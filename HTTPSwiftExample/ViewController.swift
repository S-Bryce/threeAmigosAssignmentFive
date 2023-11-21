//
//  ViewController.swift
//  HTTPSwiftExample
//
//  Created by Eric Larson on 3/30/15.
//  Copyright (c) 2015 Eric Larson. All rights reserved.
//

// This exampe is meant to be run with the python example:
//              tornado_turiexamples.py 
//              from the course GitHub repository: tornado_bare, branch sklearn_example


// if you do not know your local sharing server name try:
//    ifconfig |grep "inet "
// to see what your public facing IP address is, the ip address can be used here

// CHANGE THIS TO THE URL FOR YOUR LAPTOP
let SERVER_URL = "http://47.189.164.232:8000" // change this for your server name!!!

import UIKit
import CoreMotion

class ViewController: UIViewController, URLSessionDelegate {
    
    // MARK: Class Properties
    lazy var session: URLSession = {
        let sessionConfig = URLSessionConfiguration.ephemeral
        
        sessionConfig.timeoutIntervalForRequest = 5.0
        sessionConfig.timeoutIntervalForResource = 8.0
        sessionConfig.httpMaximumConnectionsPerHost = 1
        
        return URLSession(configuration: sessionConfig,
            delegate: self,
            delegateQueue:self.operationQueue)
    }()
    
    let operationQueue = OperationQueue()
    let motionOperationQueue = OperationQueue()
    let calibrationOperationQueue = OperationQueue()
    
    var ringBuffer = RingBuffer()
    let animation = CATransition()
    let motion = CMMotionManager()
    
    var magValue = 0.1
    var isCalibrating = false
    
    var modelType = "Logistic Classifier"
    
    var isWaitingForMotionData = false
    
    var speechToText = SpeechRecognizer()
    
    @IBOutlet weak var largeMotionMagnitude: UIProgressView!
    @IBOutlet weak var transcriptLabel: UILabel!
    
    @IBAction func InferencePressed(_ sender: Any) {
        self.speechToText.startTranscribing()
    }
    
    
    @IBAction func InferenceReleased(_ sender: Any) {
        self.speechToText.stopTranscribing()
        self.getPrediction(self.speechToText.transcript.components(separatedBy: " "), modelType: self.modelType)
        self.transcriptLabel.text = self.speechToText.transcript
        self.speechToText.resetTranscript()
    }
    

    
    @IBAction func TimerPressed(_ sender: Any) {
        self.speechToText.startTranscribing()
    }
    
    
    @IBAction func TimerReleased(_ sender: Any) {
        self.speechToText.stopTranscribing()
        self.sendFeatures(self.speechToText.transcript.components(separatedBy: " "), withLabel: "timer", modelType: self.modelType)
        self.transcriptLabel.text = self.speechToText.transcript
        self.speechToText.resetTranscript()
    }
    
    
    @IBAction func ReminderPressed(_ sender: Any) {
        self.speechToText.startTranscribing()
    }
    
    
    @IBAction func ReminderReleased(_ sender: Any) {
        self.speechToText.stopTranscribing()
        self.sendFeatures(self.speechToText.transcript.components(separatedBy: " "), withLabel: "reminder", modelType: self.modelType)
        self.transcriptLabel.text = self.speechToText.transcript
        self.speechToText.resetTranscript()
        
    }
    
    
    @IBAction func NotesPressed(_ sender: Any) {
        self.speechToText.startTranscribing()
    }
    
    
    @IBAction func NotesReleased(_ sender: Any) {
        self.speechToText.stopTranscribing()
        self.sendFeatures(self.speechToText.transcript.components(separatedBy: " "), withLabel: "notes", modelType: self.modelType)
        self.transcriptLabel.text = self.speechToText.transcript
        self.speechToText.resetTranscript()
    }
    
    
    @IBOutlet weak var modelButton: UISwitch!
    
    @IBOutlet weak var modelLabel: UILabel!
    
    @IBAction func ChangeModelType(_ sender: Any) {
        if self.modelButton.isOn {
            self.modelType = "Logistic Classifier"
            self.modelLabel.text = "Model Type: \(self.modelType)"
        }
        else {
            self.modelType = "KNN Classifier"
        }
        self.modelLabel.text = "Model Type: \(self.modelType)"
    }
    
    
    // MARK: Class Properties with Observers
    enum CalibrationStage {
        case notCalibrating
        case up
        case right
        case down
        case left
    }
    
    var dsid:Int = 0
    
    @IBAction func magnitudeChanged(_ sender: UISlider) {
        self.magValue = Double(sender.value)
    }
    
    func setDelayedWaitingToTrue(_ time:Double){
        DispatchQueue.main.asyncAfter(deadline: .now() + time, execute: {
            self.isWaitingForMotionData = true
        })
    }
    
    func setAsCalibrating(_ label: UILabel){
        label.layer.add(animation, forKey:nil)
        label.backgroundColor = UIColor.red
    }
    
    func setAsNormal(_ label: UILabel){
        label.layer.add(animation, forKey:nil)
        label.backgroundColor = UIColor.white
    }
    
    // MARK: View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        self.modelLabel.text = "Model Type: \(self.modelType)"
        // create reusable animation
        animation.timingFunction = CAMediaTimingFunction(name: CAMediaTimingFunctionName.easeInEaseOut)
        animation.type = CATransitionType.fade
        animation.duration = 0.5
        
        
        // setup core motion handlers
        
        
        dsid = 1 // set this and it will update UI
    }

    //MARK: Get New Dataset ID
    @IBAction func getDataSetId(_ sender: AnyObject) {
        // create a GET request for a new DSID from server
        let baseURL = "\(SERVER_URL)/GetNewDatasetIdMotor"
        
        let getUrl = URL(string: baseURL)
        let request: URLRequest = URLRequest(url: getUrl!)
        let dataTask : URLSessionDataTask = self.session.dataTask(with: request,
            completionHandler:{(data, response, error) in
                if(error != nil){
                    print("Response:\n%@",response!)
                }
                else{
                    let jsonDictionary = self.convertDataToDictionary(with: data)
                    
                    // This better be an integer
                    if let dsid = jsonDictionary["dsid"]{
                        self.dsid = dsid as! Int
                    }
                }
                
        })
        
        dataTask.resume() // start the task
        
    }
    
    //MARK: Comm with Server
    func sendFeatures(_ array:[String], withLabel label:String, modelType: String){
        let baseURL = "\(SERVER_URL)/AddDataPointMotor"
        let postUrl = URL(string: "\(baseURL)")
        
        // create a custom HTTP POST request
        var request = URLRequest(url: postUrl!)
        
        // data to send in body of post request (send arguments as json)
        let jsonUpload:NSDictionary = ["feature":array,
                                       "label":"\(label)",
                                       "dsid":self.dsid,
                                       "model_type": modelType] // specifies model type (mod B)
        
        
        let requestBody:Data? = self.convertDictionaryToData(with:jsonUpload)
        
        request.httpMethod = "POST"
        request.httpBody = requestBody
        
        let postTask : URLSessionDataTask = self.session.dataTask(with: request,
            completionHandler:{(data, response, error) in
                if(error != nil){
                    if let res = response{
                        print("Response:\n",res)
                    }
                }
                else{
                    let jsonDictionary = self.convertDataToDictionary(with: data)
                    
                    print(jsonDictionary["feature"]!)
                    print(jsonDictionary["label"]!)
                }

        })
        
        postTask.resume() // start the task
    }
    
    func getPrediction(_ array:[String], modelType: String){
        let baseURL = "\(SERVER_URL)/PredictOneMotor"
        let postUrl = URL(string: "\(baseURL)")
        
        // create a custom HTTP POST request
        var request = URLRequest(url: postUrl!)
        
        // data to send in body of post request (send arguments as json)
        let jsonUpload:NSDictionary = ["feature":array, "dsid":self.dsid, "model_type": modelType] // sets model type (mod B)
        
        
        let requestBody:Data? = self.convertDictionaryToData(with:jsonUpload)
        
        request.httpMethod = "POST"
        request.httpBody = requestBody
        
        let postTask : URLSessionDataTask = self.session.dataTask(with: request,
                                                                  completionHandler:{
                        (data, response, error) in
                        if(error != nil){
                            if let res = response{
                                print("Response:\n",res)
                            }
                        }
                        else{ // no error we are aware of
                            let jsonDictionary = self.convertDataToDictionary(with: data)
                            print("Response:\n",response)
                            let labelResponse = jsonDictionary["prediction"]!
                            print(labelResponse)
                            // TODO: UI Element for predicted class
                        }
                                                                    
        })
        
        postTask.resume() // start the task
    }
    
    @IBAction func makeModel(_ sender: AnyObject) {
        
        // create a GET request for server to update the ML model with current data
        let baseURL = "\(SERVER_URL)/UpdateModelMotor"
        let query = "?dsid=\(self.dsid)"
        
        let getUrl = URL(string: baseURL+query)
        let request: URLRequest = URLRequest(url: getUrl!)
        let dataTask : URLSessionDataTask = self.session.dataTask(with: request,
              completionHandler:{(data, response, error) in
                // handle error!
                if (error != nil) {
                    if let res = response{
                        print("Response:\n",res)
                    }
                }
                else{
                    let jsonDictionary = self.convertDataToDictionary(with: data)
                    
                    if let resubAcc = jsonDictionary["resubAccuracy"]{
                        print("Resubstitution Accuracy is", resubAcc)
                    }
                }
                                                                    
        })
        
        dataTask.resume() // start the task
        
    }
    
    //MARK: JSON Conversion Functions
    func convertDictionaryToData(with jsonUpload:NSDictionary) -> Data?{
        do { // try to make JSON and deal with errors using do/catch block
            let requestBody = try JSONSerialization.data(withJSONObject: jsonUpload, options:JSONSerialization.WritingOptions.prettyPrinted)
            return requestBody
        } catch {
            print("json error: \(error.localizedDescription)")
            return nil
        }
    }
    
    func convertDataToDictionary(with data:Data?)->NSDictionary{
        do { // try to parse JSON and deal with errors using do/catch block
            let jsonDictionary: NSDictionary =
                try JSONSerialization.jsonObject(with: data!,
                                              options: JSONSerialization.ReadingOptions.mutableContainers) as! NSDictionary
            
            return jsonDictionary
            
        } catch {
            
            if let strData = String(data:data!, encoding:String.Encoding(rawValue: String.Encoding.utf8.rawValue)){
                            print("printing JSON received as string: "+strData)
            }else{
                print("json error: \(error.localizedDescription)")
            }
            return NSDictionary() // just return empty
        }
    }

}





