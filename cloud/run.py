import os
import sys


'''Script to automate training job on the Google Cloud Platform through Google
   ML Engine'''


job = sys.argv[1]
file = sys.argv[2]
train_dir = sys.argv[3]
predict_dir = sys.argv[4]
custom = sys.argv[5]
batch_size = sys.argv[6]
epochs = sys.argv[7]
learning_rate = sys.argv[8]
kfold_splits = sys.argv[9]
layers = sys.argv[10:]

bucket = 'gs://nemsolutions-gcp-databucket'
output = 'output'
region = 'europe-west1'
mod = 'cloud.train'
config_file = 'config.yaml'
command = (
    ('gcloud ml-engine jobs submit training %s ' % job) +
    '--module-name %s ' % mod +
    '--config %s ' % config_file +
    '--job-dir %s/%s/%s ' % (bucket, output, job) +
    '--package-path . ' +
    '--region %s ' % region +
    '-- ' +
    '--train-file %s/%s/%s.csv ' % (bucket, train_dir, file) +
    '--predict-file %s/%s/%s.csv ' % (bucket, predict_dir, file) +
    '--job-dir %s/%s/%s ' % (bucket, output, job) +
    '--job-name %s ' % job +
    '--custom %s ' % custom +
    '--batch-size %s ' % batch_size +
    '--epochs %s ' % epochs +
    '--learning-rate %s ' % learning_rate +
    '--kfold-splits %s ' % kfold_splits +
    '--layers %s ' % layers
)

os.system(command)
