import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import {drawRect} from "./utilities.js";
import "./styles/home.css";
import "./index.css";


function AppHeader() {
  return(
    <div> 
      <h1 className="Title" style={{position: "absolute",
        top:-35}}>
        AiSL
        </h1>
      <code className="code" style={{position: "absolute", top:107}}>A web app by: Tayan Benson, Joshn Radjavitch, Nate Quero, and Nathan Wessely.</code>
    </div>
  );
}

function AboutUs() {
  return(
    <div>
      <p className="paragraph" style={{
      position: "absolute",
      top: 220,
      left: 0,
      right: 0,
      marginLeft: "auto",
      marginRight: "auto",
      textAlign: 'center',
      width: 620,
      zIndex: 0
        }}>
          AiSL was created to help combat language barriers 
          that exist in our society between American Sign Language 
          users and non-ASL users. Using our AI study tool 
          hopefully we can practice more ASL and continue to shrink the
          barriers that separate us! CLICK ON THE ABOUT US BUTTON 
          TO ENTER RECOGNITION MODE AND GET STARTED!
      </p>
    </div>
  );
}

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [show, toggleShow] = useState(false);

  const runCoco = async () => {
    const net = await tf.loadGraphModel(
      'https://tensorflowjsaislmodel.s3.us-east.cloud-object-storage.appdomain.cloud/model.json')
    
    setInterval(() => {
      detect(net);
    }, 16.7);
  };

  const detect = async (net) => {
  
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const img = tf.browser.fromPixels(video)
      const resized = tf.image.resizeBilinear(img, [640,480])
      const casted = resized.cast('int32')
      const expanded = casted.expandDims(0)
      const obj = await net.executeAsync(expanded)
      console.log(obj)

      const boxes = await obj[1].array()
      const classes = await obj[2].array()
      const scores = await obj[4].array()
      
      const ctx = canvasRef.current.getContext("2d");

      requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.8, videoWidth, videoHeight, ctx)}); 

      tf.dispose(img)
      tf.dispose(resized)
      tf.dispose(casted)
      tf.dispose(expanded)
      tf.dispose(obj)

    }
  };

  useEffect(()=>{runCoco()},[]);

  return (
    <div className="main">
      <main className="App-Header">
        <button onClick={() => toggleShow(!show)}
          className="button" 
          style={{
            position: "absolute", 
            top: 157,
            right: 0,
            left: 0
           }}>
          {show ? "Recognition Mode" : "About Us"}
        </button>
        
        {show && <Webcam
          ref={webcamRef}
          muted={true}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            top: 210,
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />}

          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              top: 210,
              left: 0,
              right: 0,
              textAlign: "center",
              zIndex: 10,
              width: 640,
              height: 480,
            }}
        />
       </main>
      <AppHeader/>
      <AboutUs/>
    </div>
  );
}

export default App;
