#!/usr/bin/python
'''Starts and runs the scikit learn server'''

import tornado.web
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from motor import motor_tornado
from pprint import PrettyPrinter

# Custom imports
from basehandler import BaseHandler
import turihandlers as th
import motorhandler as mh
import examplehandlers as eh

define("port", default=8000, help="run on the given port", type=int)

pp = PrettyPrinter(indent=4)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/[/]?", BaseHandler),
            (r"/Handlers[/]?", th.PrintHandlers),
            (r"/AddDataPoint[/]?", th.UploadLabeledDatapointHandler),
            (r"/GetNewDatasetId[/]?", th.RequestNewDatasetId),
            (r"/UpdateModel[/]?", th.UpdateModelForDatasetIdTuri),
            (r"/PredictOne[/]?", th.PredictOneFromDatasetIdTuri),
            (r"/GetExample[/]?", eh.TestHandler),
            (r"/DoPost[/]?", eh.PostHandlerAsGetArguments),
            (r"/PostWithJson[/]?", eh.JSONPostHandler),
            (r"/MSLC[/]?", eh.MSLC),
            (r"/MotorHandlers[/]?", mh.PrintHandlers),
            (r"/AddDataPointMotor[/]?", mh.UploadLabeledDatapointHandler),
            (r"/GetNewDatasetIdMotor[/]?", mh.RequestNewDatasetId),
            (r"/UpdateModelMotor[/]?", mh.UpdateModelForDatasetIdMotor),
            (r"/PredictOneMotor[/]?", mh.PredictOneFromDatasetIdMotor),
        ]

        self.handlers_string = str(handlers)

        try:
            print('=================================')
            print('====ATTEMPTING MONGO CONNECT=====')
            self.client = motor_tornado.MotorClient("localhost", 27017)
            #print(self.client.server_info())
            self.db = self.client.turidatabase

        except Exception as e:
            print('Could not initialize database connection, stopping execution')
            print(f'Error: {e}')

        self.clf = {}
        print('=================================')
        print('==========HANDLER INFO===========')
        pp.pprint(handlers)

        settings = {'debug': True}
        tornado.web.Application.__init__(self, handlers, **settings)

    def __exit__(self):
        self.client.close()

def main():
    tornado.options.parse_command_line()
    http_server = HTTPServer(Application(), xheaders=True)
    http_server.listen(options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    main()
