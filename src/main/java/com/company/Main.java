package com.company;

import javafx.util.Pair;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.commons.lang3.tuple.Pair;


public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException, InterruptedException{
        System.out.println(args[0]);
        ExecutorService es = Executors.newFixedThreadPool(6);
        Map<String,TrainResult[]> resultsC=new HashMap<String,TrainResult[]>();

        es.execute(() -> {
            try {
                DatasetTest dtt = new DatasetTest();
                Method m1 = dtt.getClass().getDeclaredMethod(args[0], Boolean.class, Boolean.class);
                TrainResult[] t = (TrainResult[]) m1.invoke(dtt, true, false);
                synchronized (resultsC) {
                    resultsC.put("AB", t);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        Thread.sleep(5000);
        es.execute(() -> {
            try {
                DatasetTest dtt = new DatasetTest();
                Method m1 = dtt.getClass().getDeclaredMethod(args[0], Boolean.class, Boolean.class);
                TrainResult[] t = (TrainResult[]) m1.invoke(dtt, false, false);
                synchronized (resultsC) {
                    resultsC.put("B", t);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        Thread.sleep(5000);
        es.execute(() -> {
            try {
                DatasetTest dtt = new DatasetTest();
                Method m1 = dtt.getClass().getDeclaredMethod(args[0], Boolean.class, Boolean.class);
                TrainResult[] t = (TrainResult[]) m1.invoke(dtt, true, true);
                synchronized (resultsC) {
                    resultsC.put("hAB", t);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        es.shutdown();
        es.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        PrintWriter pwT=new PrintWriter(new File( "Last"+".csv"));
        PrintWriter pwO=new PrintWriter(new File( args[0]+"aco-bp "+ DateTime.now().dayOfYear().get()+" " +DateTime.now().millisOfDay().get() +".csv"));
//        PrintWriter pwS=new PrintWriter(new File( "stat_cancer_aco-bp "+ DateTime.now().dayOfYear().get()+" " +DateTime.now().millisOfDay().get() +".csv"));
        TrainResult[] TRA,TRB,TRAB;
        TRA=resultsC.get("AB");
        TRB=resultsC.get("B");
        TRAB=resultsC.get("hAB");

        writeBatches(pwO, TRB);
        writeBatches(pwT,TRB);
//        writeMedian(pwS, TRA);
//        writeMedian(pwS, TRB);
//        writeMedian(pwS, TRAB);
//        pwS.flush();

        //writeBatches(pwO, TRA[0]);
        WriteTR(pwO, TRA);
        WriteTR(pwO, TRB);
        WriteTR(pwO, TRAB);
        pwO.flush();

        WriteTR(pwT, TRA);
        WriteTR(pwT, TRB);
        WriteTR(pwT, TRAB);
        pwT.flush();



    }

    private static void writeBatches(PrintWriter pw, TrainResult[] TR) {
        if (TR != null) {
            for (int i = 0; i < 1; i++) {
                for (int j = 0; j < TR[i].scores.size()-1; j++) {
                    pw.write(TR[i].scores.get(j).getKey() + ",");
                }
            }
            pw.write(TR[0].scores.get(TR[0].scores.size()-1).getKey()+"\n");
        }
    }
    @Deprecated
    private static void writeMedian(PrintWriter pwS, TrainResult[] TR) {
        if(TR!=null) {
            for (int i = 0; i < TR[0].scores.size(); i++) {
                TR[i].scores.sort((p1,p2)->p1.getValue()>p2.getValue()?1:-1);
                pwS.write(TR[i].scores.get(TR[i].scores.size()/2) + ",");
            }
            pwS.write("\n");
        }
    }

    private static void WriteTR(PrintWriter pwC, TrainResult[] TR) {
        if(TR!=null) {
            for (int i = 0; i < TR.length; i++) {
                for (int j = 0; j < TR[i].scores.size(); j++) {
                    //pwC.write(TR[i].scores.get(j).getKey() + ",");
                }
                //pwC.write("\n");
                for (int j = 0; j < TR[i].scores.size()-1; j++) {
                    pwC.write(TR[i].scores.get(j).getValue() + ",");
                }
                pwC.write(TR[i].scores.get(TR[i].scores.size()-1).getValue()+"\n");
            }
            pwC.write("\n");
        }
    }


    public static String get2DArrayPrint(double[][] matrix) {
        String output = new String();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                output = output + (String.format("%1.4f", matrix[i][j]) + "_");
            }
            output = output + "\n";
        }
        return output;
    }
}