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

public class TTTDataset {

    public static final int numFeatures=27;
    public static final int numLabels=2;
    public static final int numInstances=958;

    public List<DataSet> datasets;
    public ListDataSetIterator iterator;
    public TTTDataset(int batchSize, int seed,int numExamples) throws IOException {
        File f = new File("tic-tac-toe.data");
        double[] featureVector = new double[numFeatures * numInstances];
        double[] labelVector = new double[numLabels*numInstances];
        int[] id = new int[]{0};
        Files.lines(f.toPath()).forEach(s -> {
            String[] ss = s.split(",");
            for(int i=0;i<numFeatures/3;i++){
                switch (ss[i]){
                    case "x":
                        featureVector[id[0]*numFeatures+ i*3]=1;
                        break;
                    case "o":
                        featureVector[id[0]*numFeatures+i*3+1]=1;
                        break;
                    case "b":
                        featureVector[id[0]*numFeatures+i*3+2]=1;
                        break;
                }
            }
            labelVector[id[0] * numLabels + 0] = 0;
            labelVector[id[0] * numLabels + 1] = 0;
            String label=ss[ss.length - 1];
            switch (label){
                case "positive":
                    labelVector[id[0] * numLabels + 0]=1;
                    break;
                case "negative":
                    labelVector[id[0] * numLabels + 1]=1;
                    break;
            }
            id[0]++;
        });

        INDArray x = Nd4j.create(featureVector, new int[]{featureVector.length / numFeatures, numFeatures}, 'c');
        INDArray y = Nd4j.create(labelVector, new int[]{labelVector.length / numLabels, numLabels}, 'c');

        DataSet allData = new DataSet(x, y);

        allData.shuffle(seed);

        datasets = allData.asList();
        datasets=datasets.subList(0,numExamples);

        iterator = new ListDataSetIterator(datasets,batchSize);
        //iterator.setPreProcessor(org.nd4j.linalg.dataset.api.DataSet::scale);
    }
}
