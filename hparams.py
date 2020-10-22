import tensorflow as tf
from paths import Paths


def create_hparams_and_paths(hparams_string=None, verbose=False):
        hparams = tf.contrib.training.HParams(
                data='data/doanngocle-old',  # required
                version='GST-noempha',  # default=v.DDMM

                ################################
                # Embedding Config             #
                ################################
                p_phone_mix=0.0,  # probability of mixing phone with letter (=1 for phone only...)
                spell_oov=True,  # for transcirpt
                remove_oov=False,
                phone_vn_train='dicts/phone_vn_north',  # required for phone embedding
                phone_oov_train='',
                use_g2s=False,
                eos='#',
                punctuation='~,.*',
                special='-',
                letters='aáàạãảăắằặẵẳâấầậẫẩbcdđeéèẹẽẻêếềệễểghiíìịĩỉklmnoóòọõỏôốồộỗổơớờợỡởpqrstuúùụũủưứừựữửvxyýỳỵỹỷfjzw',
                coda_nucleus_and_semivowel=['iz', 'pc', 'nz', 'tc', 'ngz', 'kc', 'uz', 'mz', 'aa', 'ee', 'ea', 'oa',
                                            'aw', 'ie', 'uo', 'a', 'wa', 'oo', 'e', 'i', 'o', 'u', 'ow', 'uw', 'w'],
		
                ################################
                # Training Parameters          #
                ################################
                warm_start=True,
                ignore_layers=['embedding.weight', 'speakers_embedding.weight'],
                fp16_run=False,
                batch_size=32,  # 32 or 64 (if not oom)
                distributed_run=False,
                epochs=10000,
                iters_per_checkpoint=5000,
                iters_per_valid=100,
                dynamic_loss_scaling=True,
                checkpoint_path='tacotron2_pretrained.pt',

                ################################
                # Audio Preprocess             #
                ################################
                denoise=False,
                noise_frame=6,  # should=6
                trim_silence=True,
                trim_top_db=40,
                time_stretch=False,
                tempo=0.95,

                pre_emphasize=False,
                pre_emphasis=0.97,
                rescale=False,
                rescale_max=0.7,

                filter_audios=False,  # data selection
                max_num_samples_of_audio=180000,

                write_wavs_train=True,
                load_mel_from_disk=False,

                ################################
                # Feature Parameters     #
                ################################
                #sampling_rate=16000,
                sampling_rate=22050,
                filter_length=1024,
                hop_length=256,  # for 22k: 256, for 16k: 200
                win_length=1024,  # for 22k: 1024, for 16k: 800
                n_mel_channels=80,
                mel_fmin=95.0,  # for male: 55, for female: 95
                mel_fmax=7600.0,

                ################################
                # Model Parameters             #
                ################################

                # Symbols embedding
                symbols_embedding_dim=512,

                # Speaker embedding
                n_speakers=1,
                speakers_embedding_dim=128,

                # Encoder parameters
                encoder_kernel_size=5,
                encoder_n_convolutions=3,
                encoder_embedding_dim=512,

                # Decoder parameters
                n_frames_per_step=1,
                decoder_rnn_dim=1024,
                prenet_dim=256,
                p_attention_dropout=0.1,
                p_decoder_dropout=0.1,
                p_teacher_forcing=1.0,

                # Attention parameters
                attention_rnn_dim=1024,
                attention_dim=128,

                # Location Layer parameters
                attention_location_n_filters=32,
                attention_location_kernel_size=31,

                # Mel-post processing network parameters
                postnet_embedding_dim=512,
                postnet_kernel_size=5,
                postnet_n_convolutions=5,

                ################################
                # Optimization Hyperparameters #
                ################################
                use_last_lr=True,
                init_lr=1e-3,
                lr_decay=0.5,
                iter_start_decay=20000,
                iters_per_decay_lr=10000,
                final_lr=1e-5,
                eps=1e-6,
                weight_decay=1e-6,

                grad_clip_thresh=1.0,
                mask_padding=True,

                ######################
                # reference encoder  #
                ######################
                E = 512,
                ref_enc_filters = [32, 32, 64, 64, 128, 128],
                ref_enc_size = [3, 3],
                ref_enc_strides = [2, 2],
                ref_enc_pad = [1, 1],
                ref_enc_gru_size = 512 // 2,

                # Style Token Layer
                token_num = 10,
                num_heads = 8,
                n_mels = 80,
        )

        if hparams_string:
                tf.logging.info('Parsing command line hparams: %s', hparams_string)
                hparams.parse(hparams_string)

        if verbose:
                tf.logging.info('Final parsed hparams: %s', hparams.values())

        paths = Paths(hparams.data, hparams.version)

        return hparams, paths
