import threading
import queue
import numpy as np
import traceback
from tqdm import tqdm

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg
# Canonical face normalization (optional, loaded lazily)
_canonical_imports_ok = False


class StreamSDK:
    def __init__(self, cfg_pkl, data_root, **kwargs):

        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)
        
        self.default_kwargs = default_kwargs

        # ======== Canonical Face Normalization (optional) ========
        canonical_cfg = kwargs.get("canonical_cfg", default_kwargs.get("canonical_cfg", {}))
        self.canonical_cfg = canonical_cfg
        face_preprocessor = None
        canonical_template_obj = None

        if canonical_cfg.get("enabled", False):
            from core.face_preprocessing import FacePreprocessor
            from core.canonical_template import CanonicalTemplate

            preprocess_face_flag = canonical_cfg.get("preprocess_face", True)
            if preprocess_face_flag:
                face_preprocessor = FacePreprocessor(
                    mode=canonical_cfg.get("bisenet_mode", "bisenet"),
                    bisenet_path=canonical_cfg.get(
                        "bisenet_path", "checkpoints/bisenet/79999_iter.pth"
                    ),
                    device=canonical_cfg.get("bisenet_device", "cpu"),
                    fill_method=canonical_cfg.get("fill_method", "mean"),
                    blur_edge_k=canonical_cfg.get("blur_edge_k", 7),
                )
                print(f"[StreamSDK] FacePreprocessor mode: {face_preprocessor.mode}")

            # We build a temporary Source2Info-like reference for the template
            # extractor. We piggyback on the avatar_registrar's Source2Info
            # after it's created, so defer CanonicalTemplate construction.
            self._pending_canonical_cfg = canonical_cfg
            self._pending_face_preprocessor = face_preprocessor
        else:
            self._pending_canonical_cfg = None
            self._pending_face_preprocessor = None

        # ======== Core Components ========
        self.avatar_registrar = AvatarRegistrar(
            face_preprocessor=face_preprocessor,
            canonical_template=None,   # set below after source2info is ready
            **avatar_registrar_cfg,
        )

        # Build CanonicalTemplate now that Source2Info is initialised
        if self._pending_canonical_cfg is not None:
            from core.canonical_template import CanonicalTemplate
            template_path = self._pending_canonical_cfg.get(
                "template_path", "face.jpg"
            )
            canonical_template_obj = CanonicalTemplate(
                template_path=template_path,
                source2info=self.avatar_registrar.source2info,
                alpha=self._pending_canonical_cfg.get("alpha", 0.7),
                mouth_alpha=self._pending_canonical_cfg.get("mouth_alpha", 0.3),
                clamp_val=self._pending_canonical_cfg.get("clamp_val", 0.3),
            )
            self.avatar_registrar.canonical_template = canonical_template_obj
            print(
                f"[StreamSDK] Canonical normalization ENABLED — "
                f"template: {template_path}  "
                f"alpha: {self._pending_canonical_cfg.get('alpha', 0.7)}  "
                f"mouth_alpha: {self._pending_canonical_cfg.get('mouth_alpha', 0.3)}"
            )
        else:
            print("[StreamSDK] Canonical normalization DISABLED (standard mode).")

        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()

        self.wav2feat = Wav2Feat(**wav2feat_cfg)

        # Output-stage canonical state (populated in setup())
        self._canonical_enabled = self.canonical_cfg.get("enabled", False)
        self._canonical_skin_colors = []
        self._output_face_preprocessor = None


    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def _safe_put(self, q, item):
        while not self.stop_event.is_set():
            try:
                q.put(item, timeout=1)
                break
            except queue.Full:
                continue

    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        # for eye open at video end
        self.motion_stitch.set_Nd(N_d)

        # for fade in/out alpha
        if ctrl_info is None:
            ctrl_info = self.ctrl_info
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        self.ctrl_info = ctrl_info

    def setup(self, source_path, output_path, **kwargs):

        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        
        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get('smo_k_s', 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)    # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)    # for video
        self.ch_info = kwargs.get("ch_info", None)    # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)    # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")    # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        """
        ctrl_info: list or dict
            {
                fid: ctrl_kwargs
            }

            ctrl_kwargs (see motion_stitch.py):
                fade_alpha
                fade_out_keys

                delta_pitch
                delta_yaw
                delta_roll
        """

        # only hubert support online mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Unfolded Face & Extended Skin Features ========
        self.unfolded_output = kwargs.get("unfolded_output", False)
        self.remove_hair = kwargs.get("remove_hair", False)
        self.extend_skin = kwargs.get("extend_skin", False)
        
        self.face_parser = None
        self.skin_inpainter = None
        self.uv_extractor = None
        
        if self.remove_hair or self.extend_skin or self.unfolded_output:
            from core.face_parsing import FaceParser
            self.face_parser = FaceParser()
            
        if self.extend_skin:
            from core.inpainting import SkinInpainter
            self.skin_inpainter = SkinInpainter()
            
        if self.unfolded_output:
            from core.uv_extraction import UVExtractor
            self.uv_extractor = UVExtractor(self.avatar_registrar.source2info.landmark478)

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)

        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # ======== Canonical Output: Precompute skin colors ========
        # Must run AFTER avatar registration so img_rgb_lst and M_c2o_lst exist.
        self._canonical_enabled = self.canonical_cfg.get("enabled", False)
        self._canonical_skin_colors = []
        if self._canonical_enabled:
            from core.face_preprocessing import estimate_skin_color
            for frame_rgb, M_c2o in zip(
                source_info["img_rgb_lst"],
                source_info["M_c2o_lst"],
            ):
                sc = estimate_skin_color(frame_rgb, M_c2o)
                self._canonical_skin_colors.append(sc)
            print(
                f"[StreamSDK] Skin colors precomputed for "
                f"{len(self._canonical_skin_colors)} frame(s). "
                f"Sample: {self._canonical_skin_colors[0]}"
            )
            # Keep a reference to the face_preprocessor for BiSeNet output masking
            self._output_face_preprocessor = (
                self.avatar_registrar.source2info.face_preprocessor
            )

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_pbar = tqdm(desc="writer")

        # ======== Audio Feat Buffer ========
        if self.online_mode:
            # buffer: seq_frames - valid_clip_len
            self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
            assert len(self.audio_feat) == self.overlap_v2, f"{len(self.audio_feat)}"
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

        # ======== Setup Worker Threads ========
        QUEUE_MAX_SIZE = 100
        # self.QUEUE_TIMEOUT = None

        self.worker_exception = None
        self.stop_event = threading.Event()

        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.motion_stitch_worker),
            threading.Thread(target=self.warp_f3d_worker),
            threading.Thread(target=self.decode_f3d_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]

        for thread in self.thread_list:
            thread.start()

    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception as e:
            traceback.print_exc()
            return {}

    def writer_worker(self):
        try:
            self._writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _writer_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
            res_frame_rgb = item
            self.writer(res_frame_rgb, fmt="rgb")
            self.writer_pbar.update()

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self._safe_put(self.writer_queue, None)
                break
            frame_idx, render_img = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o     = self.source_info["M_c2o_lst"][frame_idx]

            # ── Canonical output post-processing ─────────────────────────────
            if self._canonical_enabled and self._canonical_skin_colors:
                from core.face_preprocessing import (
                    clean_output_frame, make_skin_background,
                )
                skin_idx   = frame_idx % len(self._canonical_skin_colors)
                skin_color = self._canonical_skin_colors[skin_idx]

                # 1. Mask the rendered face: remove rendered hair/ears,
                #    fill with skin color  (this changes the 512×512 render)
                render_img = clean_output_frame(
                    render_img,
                    skin_color,
                    face_preprocessor=self._output_face_preprocessor,
                )

                # 2. Replace source background with solid skin color so
                #    PutBack composites onto a clean background, not the
                #    original scene (which has hair / environment)
                h, w = frame_rgb.shape[:2]
                frame_rgb = make_skin_background(h, w, skin_color)
            # ─────────────────────────────────────────────────────────────────

            # ── Unfolded Face & Skin Extension ───────────────────────────────
            if self.face_parser is not None:
                # Get mask for inner face (skin, eyes, nose, lips)
                mask = self.face_parser.get_mask(render_img)
                
                if self.extend_skin and self.skin_inpainter is not None:
                    # Inpaint and extend skin texture
                    render_img = self.skin_inpainter.inpaint(render_img, mask, extend=True)
                elif self.remove_hair:
                    # Just remove hair and background
                    render_img = self.face_parser.apply_mask(render_img, mask)

            if self.unfolded_output and self.uv_extractor is not None:
                # Extract UV map and bypass putback
                res_frame_rgb = self.uv_extractor.extract_uv(render_img)
            elif self.remove_hair or self.extend_skin:
                # Option B: Render back to image space with neutral background
                # We skip putback onto original frame and just use the cleaned render
                res_frame_rgb = render_img
            else:
                # Original pipeline
                res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            # ─────────────────────────────────────────────────────────────────

            self._safe_put(self.writer_queue, res_frame_rgb)

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self._safe_put(self.putback_queue, None)
                break
            frame_idx, f_3d = item
            render_img = self.decode_f3d(f_3d)
            self._safe_put(self.putback_queue, [frame_idx, render_img])

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self._safe_put(self.decode_f3d_queue, None)
                break
            frame_idx, x_s, x_d = item
            f_s = self.source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self._safe_put(self.decode_f3d_queue, [frame_idx, f_3d])

    def motion_stitch_worker(self):
        try:
            self._motion_stitch_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _motion_stitch_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self._safe_put(self.warp_f3d_queue, None)
                break
            
            frame_idx, x_d_info, ctrl_kwargs = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            self._safe_put(self.warp_f3d_queue, [frame_idx, x_s, x_d])

    def audio2motion_worker(self):
        try:
            # self._audio2motion_worker()
            self._audio2motion_offline()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _audio2motion_offline(self):

        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue

            if item is None:
                break

            aud_feat = item

            aud_cond_all = self.condition_handler(aud_feat, 0)
            seq_frames = self.audio2motion.seq_frames
            valid_clip_len = self.audio2motion.valid_clip_len
            num_frames = len(aud_cond_all)
            idx = 0
            res_kp_seq = None
            pbar = tqdm(desc="dit")
            while idx < num_frames:
                pbar.update()
                aud_cond = aud_cond_all[idx:idx + seq_frames][None]
                if aud_cond.shape[1] < seq_frames:
                    pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                    aud_cond = np.concatenate([aud_cond, pad], 1)
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                idx += valid_clip_len

            pbar.close()
            res_kp_seq = res_kp_seq[:, :num_frames]
            res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])

            x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)

            gen_frame_idx = 0
            for x_d_info in x_d_info_list:
                frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                while not self.stop_event.is_set():
                    try:
                        self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                        break
                    except queue.Full:
                        continue
                gen_frame_idx += 1

            break

        self._safe_put(self.motion_stitch_queue, None)

        
    def _audio2motion_worker(self):
        is_end = False
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None if self.online_mode else 0
        
        global_idx = 0   # frame idx, for template
        local_idx = 0    # for cur audio_feat
        gen_frame_idx = 0
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue
            if item is None:
                is_end = True
            else:
                item_buffer = np.concatenate([item_buffer, item], 0)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                continue
            else:
                self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                # print("self.audio_feat.shape:", self.audio_feat.shape, "local_idx:", local_idx, "global_idx:", global_idx)
                aud_feat = self.audio_feat[local_idx: local_idx+seq_frames]
                real_valid_len = valid_clip_len
                if len(aud_feat) == 0:
                    break
                elif len(aud_feat) < seq_frames:
                    if not is_end:
                        # wait next chunk
                        break
                    else:
                        # final clip: pad to seq_frames
                        real_valid_len = len(aud_feat)
                        pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                        aud_feat = np.concatenate([aud_feat, pad], 0)

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                if res_kp_seq_valid_start is None:
                    # online mode, first chunk
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + real_valid_len]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                                break
                            except queue.Full:
                                continue

                        gen_frame_idx += 1

                    res_kp_seq_valid_start += real_valid_len
                
                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1] 
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

            if is_end:
                break
        
        self._safe_put(self.motion_stitch_queue, None)

    def close(self):
        # flush frames
        self._safe_put(self.audio2motion_queue, None)
        # Wait for worker threads to finish
        for thread in self.thread_list:
            thread.join()

        try:
            self.writer.close()
            self.writer_pbar.close()
        except:
            traceback.print_exc()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception
        
    def run_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        # only for hubert
        aud_feat = self.wav2feat(audio_chunk, chunksize=chunksize)
        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(aud_feat, timeout=1)
                break
            except queue.Full:
                continue
