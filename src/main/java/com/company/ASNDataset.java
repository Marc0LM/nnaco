package com.company;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;


import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class ASNDataset {

    public static final int numFeatures=5;
    public static final int numLabels=1;
    public static final int numInstances=1503;

    public List<DataSet> datasets;
    public ListDataSetIterator iterator;
    public ASNDataset(int batchSize, int seed,int numExamples) throws IOException {
        File f = new File("airfoil_self_noise.dat");
        double[] featureVector = new double[numFeatures * numInstances];
        double[] labelVector = new double[numLabels*numInstances];
        int[] id = new int[]{0};
        Files.lines(f.toPath()).forEach(s -> {
            String[] ss = s.split(" ");
            for (int i = 0; i < numFeatures; i++) {
                featureVector[id[0] * numFeatures + i] = ss[i].charAt(0)=='?'?1/512.f:(Double.valueOf(ss[i]));
            }
            labelVector[id[0]]=Double.valueOf(ss[5]);
            id[0]++;
        });

        INDArray x = Nd4j.create(featureVector, new int[]{featureVector.length / numFeatures, numFeatures}, 'c');
        INDArray y = Nd4j.create(labelVector, new int[]{labelVector.length / numLabels, numLabels}, 'c');
        for(int i=0;i<numFeatures;i++) {
            INDArray max = x.getColumn(i).max();
            INDArray min = x.getColumn(i).min();
            double toscale=max.getDouble(0)-min.getDouble(0);
            x.getColumn(i).subi(min);
            x.getColumn(i).divi(toscale);
        }
        for(int i=0;i<numLabels;i++) {
            INDArray max = y.getColumn(i).max();
            INDArray min = y.getColumn(i).min();
            double toscale=max.getDouble(0)-min.getDouble(0);
            y.getColumn(i).subi(min);
            y.getColumn(i).divi(toscale);
        }
        DataSet allData = new DataSet(x, y);

        allData.shuffle(seed);

        datasets = allData.asList();
        datasets=datasets.subList(0,numExamples);

        iterator = new ListDataSetIterator(datasets,batchSize);
        //iterator.setPreProcessor(org.nd4j.linalg.dataset.api.DataSet::scale);
    }
}