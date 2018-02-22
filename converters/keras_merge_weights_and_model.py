model_file='example/model.json'
weights_file='example/weights.h5'
combined_file='example/combined.h5'

import argparse
parser = argparse.ArgumentParser(description='Merge weights and model files into one file')
parser.add_argument('-m', action='store', dest='model_file', type=str)
parser.add_argument('-w', action='store', dest='weights_file', type=str)
parser.add_argument('-o', action='store', dest='out_file', type=str)
args = parser.parse_args()
print('input args: ', args)


from keras.models import model_from_json

# load json and create model
json_file = open(args.model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(args.weights_file)
print("Loaded model from disk")

# save model as a single .h5 file
loaded_model.save(args.out_file)

