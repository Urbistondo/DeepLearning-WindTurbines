import sys
import os

# '--scale-tier CUSTOM ' +

job = sys.argv[1]
file = sys.argv[2]
custom = sys.argv[3]
batch_size = sys.argv[4]
model_size = sys.argv[5]
epochs = sys.argv[6]
learning_rate = sys.argv[7]
kfold_splits = sys.argv[8]
layers = sys.argv[9:]

bucket = 'gs://nemsolutions-gcp-databucket'
output = 'output'
region = 'europe-west1'
mod = 'cloud.train'
config_file = 'config.yaml'
command = (('gcloud ml-engine jobs submit training %s ' % job) +
            '--module-name %s ' % mod +
            '--config %s ' % config_file +
            '--job-dir %s/%s/%s ' % (bucket, output, job) +
            '--package-path . ' +
            '--region %s ' % region +
            '-- ' +
            '--train-file %s/peaks_model/input/preprocessed/train/%s.csv ' % (bucket, file) +
            '--predict-file %s/peaks_model/input/preprocessed/train/%s.csv ' % (bucket, file) +
            '--job-dir %s/%s/%s ' % (bucket, output, job) +
            '--job-name %s ' % job +
            '--custom %s ' % custom +
            '--batch-size %s ' % batch_size +
            '--model-size %s ' % model_size +
            '--epochs %s ' % epochs +
            '--learning-rate %s ' % learning_rate +
            '--kfold-splits %s ' % kfold_splits +
            '--layers %s ' % layers)

os.system(command)
