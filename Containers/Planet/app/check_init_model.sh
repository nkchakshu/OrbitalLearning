if [ ! -f /model_zoo/banknote_authentication_model.h5 ]; then
   python3 ann_docker_model.py
fi
