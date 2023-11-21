#!/usr/bin/python

from pymongo import MongoClient
import tornado.web
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from basehandler import BaseHandler  # Add this line
import turicreate as tc
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to the screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),', '),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to the database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess = data['dsid']

        dbid = self.db.labeledinstances.insert_one(
            {"feature": fvals, "label": label, "dsid": sess}
        )
        self.write_json({"id": str(dbid),
                         "feature": [str(len(fvals)) + " Points Received",
                                     "min of: " + str(min(fvals)),
                                     "max of: " + str(max(fvals))],
                         "label": label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a is None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid']) + 1
        self.write_json({"dsid": newSessionId})

class UpdateModelForDatasetIdTuri(BaseHandler):
    def get(self):
        '''Train a new model (or update) for a given dataset ID
        '''
        dsid = self.get_int_arg("dsid", default=0)
        model_type = self.get_argument("model_type", default="default")

        data = self.get_features_and_labels_as_SFrame(dsid)

        # fit the model to the data
        turi_acc = -1  # defines Turi accuracy for later
        best_model = 'unknown'
        if len(data) > 0:
            if model_type == "xgboost":
                model = tc.classifier.create(data, target='target', model='xgboost_classifier', verbose=0)  # train model for xg (mod B part 1) boost
            else:
                model = tc.classifier.create(data, target='target', model='logistic_classifier', verbose=0)  # training

            yhat = model.predict(data)
            turi_acc = sum(yhat == data['target']) / float(len(data))

            self.clf[dsid] = model
            # save model for use later, if desired
            model.save('../models/turi_model_dsid%d' % (dsid))

        self.turi_accuracy[dsid] = turi_acc  # store Turi accuracy

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy": turi_acc})

    def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from the database
        features = []
        labels = []
        for a in self.db.labeledinstances.find({"dsid": dsid}):
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # convert to a dictionary for Turi Create
        data = {'target': labels, 'sequence': np.array(features)}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class PredictOneFromDatasetIdTuri(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))
        model_type = data.get("model_type", "default")  # change
        fvals = self.get_features_as_SFrame(data['feature'])
        dsid = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        if self.clf[dsid] == []:
            print('Loading Model From file')
            try:
                model_path = '../models/turi_model_dsid%d' % (dsid)
                self.clf[dsid] = tc.load_model('../models/turi_model_dsid%d' % (dsid))
            except:
                raise HTTPError(500)

        predLabel = self.clf[dsid].predict(fvals)
        self.write_json({"prediction": str(predLabel)})

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to a dictionary of arrays for Turi Create
        tmp = [float(val) for val in vals]
        tmp = np.array(tmp)
        tmp = tmp.reshape((1, -1))
        data = {'sequence': tmp}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

# TODO: test out the sklearn dataset responding
class UpdateModelForDatasetIdSklearn(BaseHandler):
    def get(self):
        '''Train a new model (or update) for a given dataset ID
        '''
        dsid = self.get_int_arg("dsid", default=0)

        # create feature vectors and labels from the database
        features = []
        labels = []
        for a in self.db.labeledinstances.find({"dsid": dsid}):
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # fit the model to the data
        model = KNeighborsClassifier(n_neighbors=1)
        sklearn_acc = -1  # defines sklearn acc for later
        if labels:
            model.fit(features, labels)  # training
            lstar = model.predict(features)
            self.clf[dsid] = model
            sklearn_acc = sum(lstar == labels) / float(len(labels))

            # just write this to the model files directory
            dump(model, '../models/sklearn_model_dsid%d.joblib' % (dsid))

        self.sklearn_accuracy[dsid] = sklearn_acc

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy": sklearn_acc})

class PredictOneFromDatasetIdSklearn(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        fvals = np.array(fvals).reshape(1, -1)
        dsid = data['dsid']

        # load the model (using pickle)
        if self.clf[dsid] == []:
            # load from file if needed
            print('Loading Model From DB')
            try:
                tmp = load('../models/sklearn_model_dsid%d.joblib' % (dsid))
                self.clf[dsid] = pickle.loads(tmp['model'])
            except:
                raise HTTPError(500)

        predLabel = self.clf[dsid].predict(fvals)
        self.write_json({"prediction": str(predLabel)})

# Class to compare models for sklearn and Turi (mod B part 2)
class ModelComparisonResults(BaseHandler):
    def get(self):
        dsid = self.get_int_arg("dsid", default=0)

        turi_acc = self.turi_accuracy.get(dsid, -1)
        sklearn_acc = self.sklearn_accuracy.get(dsid, -1)

        if turi_acc == -1 or sklearn_acc == -1:
            self.write_json({"error": "Unavailable: {}".format(dsid)})
            return

        # accuracies' response
        response = {
            "Turi_Create_Accuracy": turi_acc,
            "Sklearn_Accuracy": sklearn_acc,
            "DSID": dsid
        }

        self.write_json(response)

