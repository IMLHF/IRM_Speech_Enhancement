#sudo docker run --runtime=nvidia -d --name c1_irm -v /home/student/lhf/work/irm_test/c1_irm:/work -v /sdc3/tmplhf/irm-data:/data lhf/tensorflow:v1 python3 /work/run_lstm_pit_tfdata.py 0
sudo docker run --runtime=nvidia -d --name extract_tfrecord \
  -v /home/student/lhf/work/irm_test/extract_tfrecord:/work \
  -v /sdc3/tmplhf/irm-data/feature_tfrecords_utt03s_irm:/feature_tfrecords_utt03s_irm \
  -v /home/student/lhf/alldata/noise_lhf:/noise_lhf \
  -v /home/student/lhf/alldata/aishell_90_speaker:/aishell_90_speaker \
  lhf/tensorflow:v1 \
  python3 /work/run_irm.py '' extract_tfrecord
