#!/usr/bin/python

from pymongo import MongoClient
import tornado.web
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from basehandler import BaseHandler  # Add this line
from motor import motor_tornado  # Add this line

import turicreate as tc
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        """Print information about the available handlers."""
        handlers_info = {
            "PrintHandlers": str(self.application.handlers_string)
        }
        self.write_json(handlers_info)

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        """Save data point and class label to the database."""
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        vocab = ["Note", "Timer", "Reminder"]
        fvals = [0,0,0]
        for val in vals:
            if val == "Note":
                fvals[0] = 1
            if val == "Timer":
                fvals[1] = 1
            if val == "Reminder":
                fvals[2] = 1
        label = data['label']
        sess = data['dsid']

        dbid = self.db.labeledinstances.insert_one(
            {"feature": fvals, "label": label, "dsid": sess}
        )
        self.write_json({"id": str(dbid),
                         "feature": [str(len(fvals)) + " Points Received",
                                     "min of: " + str(min(fvals)),
                                     "max of: " + str(max(fvals)),
                                     vals,
                                     fvals],
                         "label": label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        """Get a new dataset ID for building a new dataset."""
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a is None:
            new_session_id = 1
        else:
            new_session_id = 1 #float(a['dsid']) + 1
        self.write_json({"dsid": new_session_id})

class UpdateModelForDatasetIdMotor(BaseHandler):
    async def get(self):
        """Train a new model (or update) for a given dataset ID."""
        dsid = self.get_int_arg("dsid", default=0)
        model_type = self.get_argument("model_type", default="default")

        data = await self.get_features_and_labels_as_SFrame(dsid)

        # fit the model to the data
        turi_acc = -1  # defines Turi accuracy for later
        best_model = 'unknown'
        if len(data) > 0:
            if model_type == "knn":
                model = tc.nearest_neighbor_classifier.create(data, target='target', verbose=0)
            else:
                model = tc.logistic_classifier.create(data, target='target', verbose=0)

            yhat = model.predict(data)
            turi_acc = sum(yhat == data['target']) / float(len(data))

            self.clf[dsid] = model
            # save model for use later, if desired
            model.save('../models/turi_model_dsid%d' % (dsid))

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy": turi_acc})

    async def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from database
        features=[]
        labels=["Note", "Timer", "Reminder"]
        print(self.db.labeledinstances.find({"dsid":dsid}))
        for a in await self.db.labeledinstances.find({"dsid":dsid}).to_list(None):
            features.append([val for val in a['feature']])

        # convert to dictionary for tc
        features = [[0,1,0], [0,0,1], [1,0,0]]
        data = {'target':labels, 'sequence':np.array(features)}
        print(data)

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class PredictOneFromDatasetIdMotor(BaseHandler):
    def post(self):
        """Predict a single data point for a given dataset ID."""
        dsid = self.get_int_arg("dsid", default=0)
        data = self.get_features_as_SFrame(self.get_arguments("vals"))
        model = self.clf.get(dsid)

        if model is not None:
            prediction = model.predict(data)
            self.write_json({"prediction": prediction[0]})
        else:
            print("broke")
            self.write_json({"error": "Model not found for DSID: {}".format(dsid)})

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to a dictionary of arrays for Turi Create
        vocab = ["Note", "Timer", "Reminder"]
        tmp = [0, 0, 0]
        for val in vals:
            if val == "Note":
                tmp[0] = 1
            if val == "Timer":
                tmp[1] = 1
            if val == "Reminder":
                tmp[2] = 1
        tmp = np.array(tmp)
        tmp = tmp.reshape((1, -1))
        data = {'sequence': tmp}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

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
