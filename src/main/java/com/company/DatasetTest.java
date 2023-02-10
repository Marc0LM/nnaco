package com.company;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class DatasetTest extends ACOBPAlgorithm {
    TrainResult[] CancerTest(Boolean ACO, Boolean hACO) {
        numFeatures = 9;
        numLabels = 2;
        numHiddens = 16;

        numExperiments = 8;
        maxBatch = 40000;
        batchSize = 8;
        TrainResult[] results = new TrainResult[numExperiments * 4];

        rate = 0.001; // learning rate

        rngSeed = 64; // random number seed for reproducibility

        numAntDivisions = 50;
        numAntPopulation = 16;

        pheEvaRate = 0.03;
        unMoveProb = 0.0;
        kernelR = 0.7;

        r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //create hidden layer
                            .activation(Activation.SOFTMAX)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            for (int fidx = 0; fidx < 4; fidx++) {
                try {
                    r = new Random(iExp);
                    DataSetIterator train = new CancerDataset(batchSize, iExp, fidx, true).iterator;
                    DataSetIterator test = new CancerDataset(batchSize, iExp, fidx, false).iterator;
                    TrainResult acc = getTrainResult(ACO, hACO, ProblemType.CLASSIFY, r, train, test);
                    System.out.println((ACO ? "A" : "") + (hACO ? "H" : "") + iExp + " " + fidx);
                    results[iExp * 4 + fidx] = acc;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] IrisTest(Boolean ACO, Boolean hACO) {
        numFeatures = IrisDataset.numFeatures;
        numLabels = IrisDataset.numLabels;
        numHiddens = 9;

        numExperiments = 4;
        maxBatch = 20000;
        batchSize = 32;
        TrainResult[] results = new TrainResult[numExperiments * 4];

        rate = 0.001; // learning rate

        rngSeed = 1; // random number seed for reproducibility

        numAntDivisions = 64;
        numAntPopulation = 16;

        pheEvaRate = 0.001;
        unMoveProb = 0.0;
        kernelR = 0.6;
        B = 0.3;
        r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //create hidden layer
                            .activation(Activation.SOFTMAX)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            for (int fidx = 0; fidx < 4; fidx++) {
                try {
                    r = new Random(iExp);
                    DataSetIterator train = new IrisDataset(batchSize, iExp, fidx, true).iterator;
                    DataSetIterator test = new IrisDataset(batchSize, iExp, fidx, false).iterator;
                    TrainResult acc = getTrainResult(ACO, hACO, ProblemType.CLASSIFY, r, train, test);
                    System.out.println((ACO ? "A" : "") + (hACO ? "H" : "") + iExp + " " + fidx);
                    results[iExp * 4 + fidx] = acc;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] TTTTest(Boolean ACO, Boolean BP) {
        numFeatures = TTTDataset.numFeatures;
        numLabels = TTTDataset.numLabels;
        numHiddens = 18;

        numExperiments = 1;
        maxBatch = 10000;
        batchSize = 32;
        TrainResult[] results = new TrainResult[numExperiments];

        rate = 0.1; // learning rate

        rngSeed = 164; // random number seed for reproducibility


        numAntDivisions = 30;
        numAntPopulation = 4;

        pheEvaRate = 0.08;
        unMoveProb = 0.0;
        kernelR = 0.8;

        Random r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //create hidden layer
                            .activation(Activation.SOFTMAX)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            try {
                DataSetIterator train = new TTTDataset(batchSize, iExp, 800).iterator;
                DataSetIterator test = new TTTDataset(batchSize, iExp, 200).iterator;
                TrainResult acc = getTrainResult(ACO, BP, ProblemType.CLASSIFY, r, train, test);
                System.out.println((ACO ? "A" : "") + (BP ? "B" : "") + iExp);
                results[iExp] = acc;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] MnistTest(Boolean ACO, Boolean BP) {
        numFeatures = 784;
        numLabels = 10;
        numHiddens = 512;

        batchSize = 32; // batch size for each epoch
        maxBatch = 80000;
        numExperiments = 1;
        TrainResult[] results = new TrainResult[numExperiments];
        rngSeed = 64; // random number seed for reproducibility
        rate = 0.001; // learning rate

        numAntDivisions = 64;
        numAntPopulation = 64;
        kernelR = 0.7;

        pheEvaRate = 0.03;
        unMoveProb = 0.00;
        r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //create hidden layer
                            .activation(Activation.SOFTMAX)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            try {
                r = new Random(rngSeed);
                DataSetIterator train = new MnistDataSetIterator(batchSize, 40000, false, true, true, iExp);
                DataSetIterator test = new MnistDataSetIterator(batchSize, 10000, false, false, true, iExp);
                TrainResult acc = getTrainResult(ACO, BP, ProblemType.CLASSIFY, r, train, test);
                System.out.println((ACO ? "A" : "") + (BP ? "B" : "") + iExp);
                for (int i = 0; i < numAntPopulation; i++) {
                    model[i].save(new File("Mnist " + i + (ACO ? "A" : "") + (BP ? "B" : "") + iExp + ".mdl"));
                }
                results[iExp] = acc;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] MPGTest(Boolean ACO, Boolean hACO) {
        numFeatures = 9;
        numLabels = 1;
        numHiddens = 9;

        numExperiments = 8;
        maxBatch = 40000;
        batchSize = 4;
        TrainResult[] results = new TrainResult[numExperiments * 4];

        rngSeed = 64; // random number seed for reproducibility
        rate = 0.001; // learning rate

        numAntDivisions = 64;
        numAntPopulation = 16;

        pheEvaRate = 0.03;
        unMoveProb = 0.00;
        B = 0.3;
        kernelR = 0.7;

        r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                            .activation(Activation.RELU)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            for (int fidx = 0; fidx < 4; fidx++) {
                try {
                    r = new Random(iExp);
                    DataSetIterator train = new MPGDataset(batchSize, rngSeed, fidx, true).iterator;
                    DataSetIterator test = new MPGDataset(batchSize, rngSeed, fidx, false).iterator;
                    System.out.println((ACO ? "A" : " ") + (hACO ? "H" : " ") + " " + iExp + " " + fidx);
                    TrainResult acc = getTrainResult(ACO, hACO, ProblemType.REGRESS, r, train, test);

                    results[iExp * 4 + fidx] = acc;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] MTLTest(Boolean ACO, Boolean hACO) {
        numFeatures = 9;
        numLabels = 12;
        numHiddens = 25;

        numExperiments = 4;
        maxBatch = 40000;
        batchSize = 4;
        TrainResult[] results = new TrainResult[numExperiments * 4];

        rngSeed = 64; // random number seed for reproducibility
        rate = 0.05; // learning rate

        numAntDivisions = 30;
        numAntPopulation = 16;

        pheEvaRate = 0.05;
        unMoveProb = 0.00;
        B = 0.3;
        kernelR = 0.7;

        r = new Random(rngSeed);
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                            .activation(Activation.RELU)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            for (int fidx = 0; fidx < 4; fidx++) {
                try {
                    r = new Random(iExp);
                    DataSetIterator train = new MTLDataset(batchSize, rngSeed, fidx, true).iterator;
                    DataSetIterator test = new MTLDataset(batchSize, rngSeed, fidx, false).iterator;

                    TrainResult acc = getTrainResult(ACO, hACO, ProblemType.REGRESS, r, train, test);
                    System.out.println((ACO ? "A" : " ") + (hACO ? "H" : " ") + " " + iExp + " " + fidx);
                    results[iExp * 4 + fidx] = acc;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] ASNTest(Boolean ACO, Boolean BP) {
        numFeatures = ASNDataset.numFeatures;
        numLabels = ASNDataset.numLabels;
        numHiddens = 8;

        numExperiments = 20;
        maxBatch = 20000;
        batchSize = 16;
        TrainResult[] results = new TrainResult[numExperiments];


        rngSeed = 64; // random number seed for reproducibility
        rate = 0.001; // learning rate

        numAntDivisions = 32;
        numAntPopulation = 2;

        pheEvaRate = 0.7;
        unMoveProb = 0.3;
        Random r = new Random();
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                            .activation(Activation.SIGMOID)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            try {
                DataSetIterator train = new ASNDataset(batchSize, rngSeed, 1200).iterator;
                DataSetIterator test = new ASNDataset(batchSize, rngSeed, 400).iterator;
                TrainResult acc = getTrainResult(ACO, BP, ProblemType.REGRESS, r, train, test);
                System.out.println((ACO ? "A" : "") + (BP ? "B" : "") + iExp);
                results[iExp] = acc;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        getES().shutdown();
        return results;
    }

    TrainResult[] ASNTestSP(Boolean ACO, Boolean BP) {
        SparkConf sconf = new SparkConf()
                .setAppName("Test spark")
                .setMaster("spark://192.168.0.102:7077")
//                .set("spark.blockManager.port", "10025")
//                .set("spark.driver.blockManager.port", "10026")
//                .set("spark.driver.port", "10027") //make all communication ports static (not necessary if you disabled firewalls, or if your nodes located in local network, otherwise you must open this ports in firewall settings)
                .set("spark.cores.max", "8")
                .set("spark.executor.memory", "512m")
                .set("spark.driver.host", "192.168.0.103")
                .setJars(new String[]{"target/NNACOSP-1.0-SNAPSHOT.jar"});
        //conf.setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(sconf);
        //JavaSparkContext sc=new JavaSparkContext("spark://192.168.217.128:7077","NNACOSP");
        int NUM_SAMPLES = 1024 * 1024;
        List<Integer> l = new ArrayList<>(NUM_SAMPLES);
        for (int i = 0; i < NUM_SAMPLES; i++) {
            l.add(i);
        }
        //sc.parallelize(l).filter(i->i>0);
        System.out.println("11111111111111111111111111");
        JavaRDD<MultiLayerNetwork> modelRDD = sc.parallelize(model.)
        long count = sc.parallelize(l).filter(new Function<Integer, Boolean>() {
            @Override
            public Boolean call(Integer integer) throws Exception {
                double x = Math.random();
                double y = Math.random();
                return x * x + y * y < 1;
            }
        }).count();
        System.out.println("Pi is roughly " + 4.0 * count / NUM_SAMPLES);

        numFeatures = ASNDataset.numFeatures;
        numLabels = ASNDataset.numLabels;
        numHiddens = 8;

        numExperiments = 20;
        maxBatch = 20000;
        batchSize = 16;
        TrainResult[] results = new TrainResult[numExperiments];


        rngSeed = 64; // random number seed for reproducibility
        rate = 0.001; // learning rate

        numAntDivisions = 32;
        numAntPopulation = 2;

        pheEvaRate = 0.7;
        unMoveProb = 0.3;
        Random r = new Random();
        initMatModel();

        for (int i = 0; i < numAntPopulation; i++) {
            conf[i] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create hidden layer
                            .activation(Activation.SIGMOID)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[i] = new MultiLayerNetwork(conf[i]);
            model[i].init();
        }

        for (int iExp = 0; iExp < numExperiments; iExp++) {
            try {
                DataSetIterator train = new ASNDataset(batchSize, rngSeed, 1200).iterator;
                DataSetIterator test = new ASNDataset(batchSize, rngSeed, 400).iterator;
                TrainResult acc = getTrainResult(ACO, BP, ProblemType.REGRESS, r, train, test);
                System.out.println((ACO ? "A" : "") + (BP ? "B" : "") + iExp);
                results[iExp] = acc;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        getES().shutdown();
        return results;
    }


}
