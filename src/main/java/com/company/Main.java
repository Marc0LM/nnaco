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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Console;

import java.io.IOException;

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
        //number of rows and columns in the input pictures
        final int numRows = 2;
        final int numColumns = 3;
        int outputNum = 2; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 1; // number of epochs to perform
        double rate = 0.0015; // learning rate

        //Get the DataSetIterators:
        DataSetIterator t= new
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(rate * 0.05) // regularize learning model
                .list()
                .layer(new DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numColumns)
                        .nOut(5)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nOut(outputNum)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));  //print the score with every iteration
        for(int i=0;i<15;i++) {

            System.out.println("Train model....");
            //model.fit(mnistTrain, numEpochs);
            Object o = model.getLayers()[0].getParam("W");

            System.out.println("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);

            System.out.println(eval.accuracy());
            Console.println("****************Example finished********************");
        }

    }
}
