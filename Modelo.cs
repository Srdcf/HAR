using UnityEngine;
using Unity.Barracuda;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using Unity.Mathematics;
using System.Globalization;
using System.IO;

public class Modelo : MonoBehaviour
{
    public NNModel modelFile; // Reference to the Barracuda model asset

    private IWorker worker;
   

    void Start()
    {
         var model = ModelLoader.Load(modelFile);

        // Create a worker to run inference
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }


    public double[] RunInference(double[] inputScaled)
    {
         
          // Load the model
        //Debug.Log("Input Scaled: " +  string.Join(", ",inputScaled));

        float[] inputScaledFloat = new float[inputScaled.Length];
for (int i = 0; i < inputScaled.Length; i++)
{
    inputScaledFloat[i] = (float)inputScaled[i];
} 



Tensor inputTensor = new Tensor(1,17, inputScaledFloat);
       
       Debug.Log("Input Tensor: " + TensorToString(inputTensor));
        // Run inference
        worker.Execute(inputTensor);
        // Get output tensor
        Tensor outputTensor = worker.PeekOutput();
        Debug.Log("output Tensor: " + TensorToString(outputTensor));
        // Extract the output data
        float[] outputArray = outputTensor.ToReadOnlyArray();
        double[] outputData = Array.ConvertAll(outputArray, f => (double)f);

        // Clean up
        inputTensor.Dispose();
        outputTensor.Dispose();

        return outputData;
    }

string TensorToString(Tensor tensor)
{
    StringBuilder sb = new StringBuilder();
    sb.Append("[");

    for (int i = 0; i < tensor.length; i++)
    {
        if (i > 0)
        {
            sb.Append(", ");
        }
        sb.Append(tensor[i].ToString("0.########").Replace(',', '.'));
    }

    sb.Append("]");
    return sb.ToString();
}
 

    void OnDestroy()
    {
        // Dispose the worker when it's no longer needed
        if (worker != null)
            worker.Dispose();
    }
}

