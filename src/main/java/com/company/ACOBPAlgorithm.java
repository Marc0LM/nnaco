package com.company;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.joda.time.DateTime;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class ACOBPAlgorithm {

    ExecutorService es = Executors.newFixedThreadPool(12);

    ExecutorService getES() {
        return es;
    }

    Random r;

    double B = 0.1;

    int numFeatures = 9;
    int numLabels = 2;
    int numHiddens = 6;

    int minBatch = 4;
    volatile int maxBatch = 32;
    int sampleI = 10;

    int batchSize = 4; // batch size for each epoch
    int rngSeed = 64; // random number seed for reproducibility
    public static int numExperiments = 100; // number of epochs to perform
    double rate = 0.002; // learning rate
    MultiLayerNetwork[] model, modelf;
    MultiLayerConfiguration[] conf;

    int numAntDivisions = 32;
    int numAntPopulation = 2;
    double pheEvaRate = 0.28;
    double unMoveProb = 0.12;

    double kernelR = 0.001;

    double[][][] w0, b0, w1, b1;
    double[][][] w0p, b0p, w1p, b1p;
    int[][][] w0c, b0c, w1c, b1c;

    double[][][] w0f, b0f, w1f, b1f;
    double[][][] w0pf, b0pf, w1pf, b1pf;
    int[][][] w0cf, b0cf, w1cf, b1cf;

    double initialP = 0;

    public enum ProblemType {
        CLASSIFY,
        REGRESS;
    }

    public void initMatModel() {
        w0 = new double[numFeatures][numHiddens][numAntDivisions];
        b0 = new double[1][numHiddens][numAntDivisions];
        w1 = new double[numHiddens][numLabels][numAntDivisions];
        b1 = new double[1][numLabels][numAntDivisions];
        w0p = new double[numFeatures][numHiddens][numAntDivisions];
        b0p = new double[1][numHiddens][numAntDivisions];
        w1p = new double[numHiddens][numLabels][numAntDivisions];
        b1p = new double[1][numLabels][numAntDivisions];
        w0c = new int[numAntPopulation][numFeatures][numHiddens];
        b0c = new int[numAntPopulation][1][numHiddens];
        w1c = new int[numAntPopulation][numHiddens][numLabels];
        b1c = new int[numAntPopulation][1][numLabels];
        w0f = new double[numFeatures][numHiddens][numAntDivisions];
        b0f = new double[1][numHiddens][numAntDivisions];
        w1f = new double[numHiddens][numLabels][numAntDivisions];
        b1f = new double[1][numLabels][numAntDivisions];
        w0pf = new double[numFeatures][numHiddens][numAntDivisions];
        b0pf = new double[1][numHiddens][numAntDivisions];
        w1pf = new double[numHiddens][numLabels][numAntDivisions];
        b1pf = new double[1][numLabels][numAntDivisions];
        w0cf = new int[numAntPopulation][numFeatures][numHiddens];
        b0cf = new int[numAntPopulation][1][numHiddens];
        w1cf = new int[numAntPopulation][numHiddens][numLabels];
        b1cf = new int[numAntPopulation][1][numLabels];
        model = new MultiLayerNetwork[numAntPopulation];
        conf = new MultiLayerConfiguration[numAntPopulation];
    }

    public void initModelValues(Random r, double initialP) {
        for (int k = 0; k < numAntDivisions; k++) {
            //init values
            initVPMatValue(initialP, k, numHiddens, numFeatures, w0, w0p, b0, b0p);
            initVPMatValue(initialP, k, numLabels, numHiddens, w1, w1p, b1, b1p);
        }
        Forward();

        for (int antI = 0; antI < numAntPopulation; antI++) {
            for (int j = 0; j < numFeatures; j++) {
                for (int k = 0; k < numHiddens; k++) {
                    double sumPhe = numAntDivisions / (float) (numFeatures + numHiddens + numLabels);
                    ProbSelect(numAntDivisions, w0p, w0c, antI, j, k, sumPhe);
                }
            }

            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < numHiddens; k++) {
                    double sumPhe = numAntDivisions / (float) (numFeatures + numHiddens + numLabels);
                    ProbSelect(numAntDivisions, b0p, b0c, antI, j, k, sumPhe);
                }
            }

            for (int j = 0; j < numHiddens; j++) {
                for (int k = 0; k < numLabels; k++) {
                    double sumPhe = numAntDivisions / (float) (numFeatures + numHiddens + numLabels);
                    ProbSelect(numAntDivisions, w1p, w1c, antI, j, k, sumPhe);
                }
            }

            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < numLabels; k++) {
                    double sumPhe = numAntDivisions / (float) (numFeatures + numHiddens + numLabels);
                    ProbSelect(numAntDivisions, b1p, b1c, antI, j, k, sumPhe);
                }
            }
        }

        SyncToModel();
    }

    public void initVPMatValue(double initialP, int k, int numHiddens, int numFeatures, double[][][] w0, double[][][] w0p, double[][][] b0, double[][][] b0p) {
        for (int j = 0; j < numHiddens; j++) {
            for (int i = 0; i < numFeatures; i++) {//TODO distribution?
                w0[i][j][k] = k / (float) numAntDivisions * 2 - 1;
                //w0[i][j][k] = r.nextGaussian();
                w0p[i][j][k] = initialP;
            }
            b0[0][j][k] = k / (float) numAntDivisions * 2 - 1;
            //b0[0][j][k] = r.nextGaussian();
            b0p[0][j][k] = initialP;
        }
    }

    public void syncParamW(int jj, int kk, double[][][] param, int[][] ints, double[][] w) {
        for (int j = 0; j < jj; j++) {
            for (int k = 0; k < kk; k++) {
                param[j][k][ints[j][k]] = w[j][k];
            }
        }
    }

    public void syncParamB(int kk, double[][][] param, int[][] ints, double[] b) {
        for (int k = 0; k < kk; k++) {
            param[0][k][ints[0][k]] = b[k];
        }
    }

    public void pheEva(int ii, int jj, double pheEvaRate, double[][][] matP, double pheMin, double pheMax, int k, double[][] doubles) {
        for (int j = 0; j < jj; j++) {
            for (int i = 0; i < ii; i++) {
                matP[i][j][k] *= (1 - pheEvaRate);
                matP[i][j][k] = Math.max(matP[i][j][k], pheMin);
                //matP[i][j][k] = Math.min(matP[i][j][k], pheMax);
            }
            doubles[j][k] *= (1 - pheEvaRate);
            doubles[j][k] = Math.max(doubles[j][k], pheMin);
            //doubles[j][k] = Math.min(doubles[j][k], pheMax);
        }
    }

    public void updateBestPhe(int kk, int ll, double[][][] matP, double deltaP, double pheMax, int[][] choice, double[][][] w) {
        for (int k = 0; k < kk; k++) {
            for (int l = 0; l < ll; l++) {
                for (int i = 0; i < numAntDivisions; i++) { // TODO kernel function?
                    double r = Math.abs(w[k][l][choice[k][l]] - w[k][l][i]);
                    if (r < kernelR) {
                        matP[k][l][i] += deltaP * Math.pow((1 - r / kernelR), 16);
                        if (matP[k][l][i] > pheMax) {
                            matP[k][l][i] = pheMax;
                        }
                    }
                }
            }
        }
    }

    @Deprecated
    public void ProbSelectSync(int layer, String param, int numAntDivisions, MultiLayerNetwork[] model,
                               double[][][] b1, double[][][] b1p, int[][][] b1c, int antI, int j, int k, double sumPhe) {
        ProbSelect(numAntDivisions, b1p, b1c, antI, j, k, sumPhe);
    }

    public void ProbSelect(int numAntDivisions, double[][][] b1p, int[][][] b1c, int antI, int j, int k, double sumPhe) {
        double prob = r.nextDouble();
        double cumulativeProbability = 0;
        for (int l = 0; l < numAntDivisions; l++) {
            cumulativeProbability += b1p[j][k][l];
            if (prob * sumPhe <= cumulativeProbability) {
                b1c[antI][j][k] = l;
                break;
            }
        }
    }

    public void SyncToModel() {
        for (int antI = 0; antI < numAntPopulation; antI++) {
            INDArray tl;
            tl = model[antI].getLayer(0).getParam("W");
            for (int k = 0; k < numHiddens; k++) {
                for (int j = 0; j < numFeatures; j++) {
                    //tl.putScalar(j, k, w0[j][k][w0c[antI][j][k]]);
                    tl.data().put(k * numFeatures + j, w0[j][k][w0c[antI][j][k]]);
                }
            }
            tl = model[antI].getLayer(0).getParam("b");
            for (int k = 0; k < numHiddens; k++) {
                for (int j = 0; j < 1; j++) {
                    //tl.putScalar(j, k, b0[j][k][b0c[antI][j][k]]);
                    tl.data().put(k * 1 + j, b0[j][k][b0c[antI][j][k]]);
                }
            }
            tl = model[antI].getLayer(1).getParam("W");
            for (int k = 0; k < numLabels; k++) {
                for (int j = 0; j < numHiddens; j++) {
                    //tl.putScalar(j, k, w1[j][k][w1c[antI][j][k]]);
                    tl.data().put(k * numHiddens + j, w1[j][k][w1c[antI][j][k]]);
                }
            }
            tl = model[antI].getLayer(1).getParam("b");
            for (int k = 0; k < numLabels; k++) {
                for (int j = 0; j < 1; j++) {
                    //tl.putScalar(j, k, b1[j][k][b1c[antI][j][k]]);
                    tl.data().put(k * 1 + j, b1[j][k][b1c[antI][j][k]]);
                }
            }
        }
    }

    public void Forward() {
        ArrayCopy3D(w0, w0f);
        ArrayCopy3D(b0, b0f);
        ArrayCopy3D(w1, w1f);
        ArrayCopy3D(b1, b1f);
        ArrayCopy3D(w0p, w0pf);
        ArrayCopy3D(b0p, b0pf);
        ArrayCopy3D(w1p, w1pf);
        ArrayCopy3D(b1p, b1pf);
        ArrayCopy3D(w0c, w0cf);
        ArrayCopy3D(b0c, b0cf);
        ArrayCopy3D(w1c, w1cf);
        ArrayCopy3D(b1c, b1cf);
    }

    public void Back() {
        ArrayCopy3D(w0f, w0);
        ArrayCopy3D(b0f, b0);
        ArrayCopy3D(w1f, w1);
        ArrayCopy3D(b1f, b1);
        ArrayCopy3D(w0pf, w0p);
        ArrayCopy3D(b0pf, b0p);
        ArrayCopy3D(w1pf, w1p);
        ArrayCopy3D(b1pf, b1p);
        ArrayCopy3D(w0cf, w0c);
        ArrayCopy3D(b0cf, b0c);
        ArrayCopy3D(w1cf, w1c);
        ArrayCopy3D(b1cf, b1c);
    }

    public TrainResult getTrainResult(Boolean ACO, Boolean hACO, ProblemType pt, Random r, DataSetIterator train, DataSetIterator test) throws Exception {
        int ACObatch = 0;
        int bestAnt = 0;
        int tbestAnt = 0;
        double best = 1;
        double initialScore = 0;
        double[] tScore = new double[numAntPopulation];
        double acc = 0;
        double maxacc = 0;
        double tacc = 0;
        TrainResult TR = new TrainResult(maxBatch / sampleI);
        int batch = 0;
        initialP = 1 / (float) (numFeatures + numHiddens + numLabels);

        //init values
        initModelValues(r, initialP);

        boolean isBetter = false;
        int startTimeMillis = DateTime.now().millisOfDay().get();
        double[] sums = new double[maxBatch];
        while (true) {
            if (!train.hasNext()) {
                train.reset();
            }
            final DataSet d = train.next();

            //ACO
            if (ACO) {
                //prob select
                if (true) {
                    for (int antI = 0; antI < numAntPopulation; antI++) {
                        for (int j = 0; j < numFeatures; j++) {
                            for (int k = 0; k < numHiddens; k++) {
                                double sumPhe = 0;
                                for (int l = 0; l < numAntDivisions; l++) {
                                    sumPhe += w0p[j][k][l];
                                }
                                if (Math.random() < 1 - unMoveProb) {
                                    ProbSelect(numAntDivisions, w0p, w0c, antI, j, k, sumPhe);
                                }
                            }
                        }
                        for (int j = 0; j < 1; j++) {
                            for (int k = 0; k < numHiddens; k++) {
                                double sumPhe = 0;
                                for (int l = 0; l < numAntDivisions; l++) {
                                    sumPhe += b0p[j][k][l];
                                }
                                if (Math.random() < 1 - unMoveProb) {
                                    ProbSelect(numAntDivisions, b0p, b0c, antI, j, k, sumPhe);
                                }
                            }
                        }
                        for (int j = 0; j < numHiddens; j++) {
                            for (int k = 0; k < numLabels; k++) {
                                double sumPhe = 0;
                                for (int l = 0; l < numAntDivisions; l++) {
                                    sumPhe += w1p[j][k][l];
                                }
                                if (Math.random() < 1 - unMoveProb) {
                                    ProbSelect(numAntDivisions, w1p, w1c, antI, j, k, sumPhe);
                                }
                            }
                        }
                        for (int j = 0; j < 1; j++) {
                            for (int k = 0; k < numLabels; k++) {
                                double sumPhe = 0;
                                for (int l = 0; l < numAntDivisions; l++) {
                                    sumPhe += b1p[j][k][l];
                                }
                                if (Math.random() < 1 - unMoveProb) {
                                    ProbSelect(numAntDivisions, b1p, b1c, antI, j, k, sumPhe);
                                }
                            }
                        }
                    }
                }
                //sync to models
                SyncToModel();

                //choose updateBestPhe ant
                List<CompletableFuture> c = IntStream.range(0, numAntPopulation).mapToObj(i -> CompletableFuture.supplyAsync(() -> tScore[i] = model[i].score(d), getES())).collect(Collectors.toList());
                c.stream().map(CompletableFuture::join).collect(Collectors.toList());
                double tBest = Double.MAX_VALUE;
                for (int i = 0; i < numAntPopulation; i++) {
                    if (tScore[i] < tBest) {
                        tBest = tScore[i];
                        tbestAnt = i;
                    }
                }
                if (batch == 0) {
                    initialScore = tBest;
                }
                if (tBest < best) {//worse-fallback*
                    isBetter = true;
                    best = tBest;
                    bestAnt = tbestAnt;
                } else {
                    isBetter = false;
                }
                isBetter = true;
                best = tBest * B + best * (1 - B);
                if (!hACO) {
                    pheEvaRate = 0.01;
                    kernelR = 0.01;
                } else {
                    pheEvaRate = (1 / Math.pow(1.01, batch / 10) + 0.001) * 0.1;
                    unMoveProb = 0.99 - (Math.pow(0.99, batch / 10));
                }
                if (isBetter) {
                    for (int i = 0; i < numAntPopulation; i++) {
                        //update phe
                        double deltaP = 1 / Math.pow(tScore[i] / best, 2);//smooth factor *
                        double pheMax = 1000 / best / pheEvaRate;//todo
                        pheMax = Double.MAX_VALUE;
                        double tpheE = pheEvaRate;
                        //phe eva
                        double pheMin = deltaP / (2 * numAntDivisions);//todo
                        //pheMin = Double.MIN_VALUE;
                        if (i == 0) {
                            for (int k = 0; k < numAntDivisions; k++) {
                                pheEva(numFeatures, numHiddens, tpheE, w0p, pheMin, pheMax, k, b0p[0]);
                                pheEva(numHiddens, numLabels, tpheE, w1p, pheMin, pheMax, k, b1p[0]);
                            }
                        }

                        //updateBestPhe
                        updateBestPhe(numFeatures, numHiddens, w0p, deltaP, pheMax, w0c[i], w0);
                        updateBestPhe(1, numHiddens, b0p, deltaP, pheMax, b0c[i], b0);
                        updateBestPhe(numHiddens, numLabels, w1p, deltaP, pheMax, w1c[i], w1);
                        updateBestPhe(1, numLabels, b1p, deltaP, pheMax, b1c[i], b1);
                    }
                    //ACObatch++;
                    //Forward();
                    //epoch--;
                } else {
                    Back();
                    SyncToModel();
                }
                //BP and sync params
                //if (BP) {
                for (int i = 0; i < numAntPopulation; i++) {
                    model[i].fit(d);
                    double[][] tw0, tw1;
                    double[] tb0, tb1;
                    tw0 = model[i].getLayer(0).getParam("W").toDoubleMatrix();
                    tb0 = model[i].getLayer(0).getParam("b").toDoubleVector();
                    //if (pt.equals(ProblemType.REGRESS)) {
                    //double[] ttw1 = model[i].getLayer(1).getParam("W").toDoubleVector();
                    //tw1 = new double[numHiddens][1];
                    //for (int j = 0; j < numHiddens; j++) {
                    //tw1[j][0] = ttw1[j];
                    //}
                    //} else {
                    tw1 = model[i].getLayer(1).getParam("W").toDoubleMatrix();
                    //}
                    tb1 = model[i].getLayer(1).getParam("b").toDoubleVector();
                    syncParamW(numFeatures, numHiddens, w0, w0c[i], tw0);
                    syncParamB(numHiddens, b0, b0c[i], tb0);
                    syncParamW(numHiddens, numLabels, w1, w1c[i], tw1);
                    syncParamB(numLabels, b1, b1c[i], tb1);
                }
                //Forward();
//                    double[] grad=model[bestAnt].gradient().gradient().data().asDouble();
//                    double sum=0;
//                    for(int i=0;i<grad.length;i++){
//                        sum+=grad[i];
//                    }
//                    sums[batch]=sum;
//                    pheEvaRate=0.01*sum;
                //}
            } else {
                model[bestAnt].fit(d);
            }
            batch++;
            if (batch % 10 == 0) {//TODO
                switch (pt) {
                    case CLASSIFY:
                        //tacc=model[bestAnt].evaluate(test).f1();
                        tacc = model[bestAnt].evaluate(test).accuracy();
                        break;
                    case REGRESS:
                        tacc = model[bestAnt].evaluateRegression(test).averagerelativeSquaredError();
                        break;
                }
                acc = tacc;
                TR.addScore(acc, batch);
                //System.out.println((ACO ? "A" : "") + (hACO ? "h" : "") + " " + ACObatch + " " + batch + " " + tacc);
            }
            if (batch >= maxBatch) {
                break;
            }
        }

        System.out.println(DateTime.now().toString("HH:mm:ss"));
        return TR;
    }

    public void ArrayCopy3D(int[][][] int1, int[][][] int2) {
        for (int i = 0; i < int1.length; i++) {
            for (int j = 0; j < int1[i].length; j++) {
                int2[i][j] = int1[i][j].clone();
            }
        }
    }

    public void ArrayCopy3D(double[][][] double1, double[][][] double2) {
        for (int i = 0; i < double1.length; i++) {
            for (int j = 0; j < double1[i].length; j++) {
                double2[i][j] = double1[i][j].clone();
            }
        }
    }

    @Deprecated
    public TrainResult getTrainResultRegress(Boolean ACO, Boolean BP, MultiLayerConfiguration[] conf, Random r, DataSetIterator train, DataSetIterator test) throws ExecutionException, InterruptedException {

        int bestAnt = 0;
        double acc = Double.MAX_VALUE;
        double tacc = 1;
        TrainResult TR = new TrainResult(maxBatch);
        int epoch = 0;
        initialP = 1 / (float) (numFeatures + numHiddens + numLabels);
        //init values
        if (ACO || true) {
            initModelValues(r, initialP);
        } else {
            conf[0] = new NeuralNetConfiguration.Builder()
                    .seed(r.nextInt()) //include a random seed for reproducibility
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.NORMAL)
                    .updater(new Sgd(rate))
                    .list()
                    .layer(new DenseLayer.Builder() //create the first input layer.
                            .nIn(numFeatures)
                            .nOut(numHiddens)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) //create hidden layer
                            .activation(Activation.SIGMOID)
                            .nOut(numLabels)
                            .build())
                    .build();
            model[0] = new MultiLayerNetwork(conf[0]);
            model[0].init();
        }

        while (true) {
            if (!train.hasNext()) {
                train.reset();
            }
            final DataSet d = train.next();
            //ACO prob select
            if (ACO) {
                for (int antI = 0; antI < numAntPopulation; antI++) {
                    for (int j = 0; j < numFeatures; j++) {
                        for (int k = 0; k < numHiddens; k++) {
                            double sumPhe = 0;
                            for (int l = 0; l < numAntDivisions; l++) {
                                sumPhe += w0p[j][k][l];
                            }
                            if (Math.random() < 1 - unMoveProb) {
                                ProbSelectSync(0, "W", numAntDivisions, model, w0, w0p, w0c, antI, j, k, sumPhe);
                            }
                        }
                    }
                    for (int j = 0; j < 1; j++) {
                        for (int k = 0; k < numHiddens; k++) {
                            double sumPhe = 0;
                            for (int l = 0; l < numAntDivisions; l++) {
                                sumPhe += b0p[j][k][l];
                            }
                            if (Math.random() < 1 - unMoveProb) {
                                ProbSelectSync(0, "b", numAntDivisions, model, b0, b0p, b0c, antI, j, k, sumPhe);
                            }
                        }
                    }
                    for (int j = 0; j < numHiddens; j++) {
                        for (int k = 0; k < numLabels; k++) {
                            double sumPhe = 0;
                            for (int l = 0; l < numAntDivisions; l++) {
                                sumPhe += w1p[j][k][l];
                            }
                            if (Math.random() < 1 - unMoveProb) {
                                ProbSelectSync(1, "W", numAntDivisions, model, w1, w1p, w1c, antI, j, k, sumPhe);
                            }
                        }
                    }
                    for (int j = 0; j < 1; j++) {
                        for (int k = 0; k < numLabels; k++) {
                            double sumPhe = 0;
                            for (int l = 0; l < numAntDivisions; l++) {
                                sumPhe += b1p[j][k][l];
                            }
                            if (Math.random() < 1 - unMoveProb) {
                                ProbSelectSync(1, "b", numAntDivisions, model, b1, b1p, b1c, antI, j, k, sumPhe);
                            }
                        }
                    }
                }
                //choose updateBestPhe ant
                double best = Double.MAX_VALUE;
                double[] tScore = new double[numAntPopulation];
                List<CompletableFuture<Double>> c = IntStream.range(0, numAntPopulation).mapToObj(i -> CompletableFuture.supplyAsync(() -> tScore[i] = model[i].score(d), getES())).collect(Collectors.toList());
                c.stream().map(CompletableFuture::join).collect(Collectors.toList());
                for (int i = 0; i < numAntPopulation; i++) {
                    if (tScore[i] < best) {
                        best = tScore[i];
                        bestAnt = i;
                    }
                }

                //BP  sync params
                for (int i = 0; i < numAntPopulation; i++) {
                    if (BP) {
                        if (i == bestAnt) {
                            final int ii = i;
                            final double[][] tw0, tw1;
                            final double[] tb0, tb1;
                            tw0 = model[ii].getLayer(0).getParam("W").toDoubleMatrix();
                            tb0 = model[ii].getLayer(0).getParam("b").toDoubleVector();
                            tw1 = model[ii].getLayer(1).getParam("W").toDoubleMatrix();
                            tb1 = model[ii].getLayer(1).getParam("b").toDoubleVector();
                            List<Future> f = new ArrayList<>(numAntPopulation);
                            model[i].fit(d);
                            f.add(getES().submit(() -> syncParamW(numFeatures, numHiddens, w0, w0c[ii], tw0)));
                            f.add(getES().submit(() -> syncParamB(numHiddens, b0, b0c[ii], tb0)));
                            f.add(getES().submit(() -> syncParamW(numHiddens, numLabels, w1, w1c[ii], tw1)));
                            f.add(getES().submit(() -> syncParamB(numLabels, b1, b1c[ii], tb1)));
                            for (int ff = 0; ff < 4; ff++) {
                                f.get(ff).get();
                            }
                        }
                    }
                }
                //update phe

                double deltaP = 1 / best;
                double pheMax = 1 / (pheEvaRate * best);
                //phe eva
                double pheMin = pheMax / (20000 * numAntDivisions);
                for (int k = 0; k < numAntDivisions; k++) {
                    pheEva(numFeatures, numHiddens, pheEvaRate, w0p, pheMin, pheMax, k, b0p[0]);
                    pheEva(numHiddens, numLabels, pheEvaRate, w1p, pheMin, pheMax, k, b1p[0]);
                }
                //updateBestPhe
                updateBestPhe(numFeatures, numHiddens, w0p, deltaP, pheMax, w0c[bestAnt], w0);
                updateBestPhe(1, numHiddens, b0p, deltaP, pheMax, b0c[bestAnt], b0);
                updateBestPhe(numHiddens, numLabels, w1p, deltaP, pheMax, w1c[bestAnt], w1);
                updateBestPhe(1, numLabels, b1p, deltaP, pheMax, b1c[bestAnt], b1);

            } else {
                if (BP) {
                    model[0].fit(d);
                }
            }

            epoch++;
            if (epoch % sampleI == 0) {//TODO
                tacc = model[bestAnt].evaluateRegression(test).averagerelativeSquaredError();
                TR.addScore(tacc, epoch);
            }

            if (epoch >= maxBatch) {
                break;
            }
        }

        return TR;
    }
}
