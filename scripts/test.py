import pickle
import os

encoder = pickle.load(open(os.path.join(os.getcwd(),'models/svc_target_encoder.pkl'),"rb"))
print(encoder.classes_)
