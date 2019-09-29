package com.company;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.example.mnist.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Console;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1023; // random number seed for reproducibility
        int numEpochs = 1; // number of epochs to perform
        double rate = 0.15; // learning rate

        INDArray t= Nd4j.zeros(10,10).addi(0);
        for(int i=0;i<3;i++){
            for(int j=0;j<4;j++){
                t.putScalar(i,j,i*10+j);
            }
        }
        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(rate * 0.005) // regularize learning model
                .list()
                .layer(new DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                        .activation(Activation.SIGMOID)
                        .nOut(outputNum)
                        .build())
                .build();
        MultiLayerNetwork model;
        try {
            File f = new File("01.mdl");
            model = MultiLayerNetwork.load(f, false);
            System.out.println("Load model.... ");
        }
        catch (FileNotFoundException fnfe) {
            model = new MultiLayerNetwork(conf);
            model.init();
            model.save(new File("0.mdl"));
            System.out.println("Save model.... ");
        }

        System.out.println("Train model....");

        double b0=0;
        for(int i=0;i<150;i++) {

            //model.getLayer(0).getParam("b").putScalar(0, 0,b0);
            model.fit(mnistTrain, numEpochs);

            System.out.println("Evaluate model.... "+i);
            Evaluation eval = model.evaluate(mnistTest);

            System.out.println(eval.accuracy());
        }

    }
}
