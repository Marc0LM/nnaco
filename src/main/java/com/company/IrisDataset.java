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

public class IrisDataset {

    public static final int numFeatures=4;
    public static final int numLabels=3;
    public static final int numInstances=150;

    public List<DataSet> datasets;
    public ListDataSetIterator iterator;
    public IrisDataset(int batchSize, int seed,int foldidx,Boolean train) throws IOException {
        File f = new File("iris.data");
        double[] featureVector = new double[numFeatures * numInstances];
        double[] labelVector = new double[numLabels*numInstances];
        int[] id = new int[]{0};
        Files.lines(f.toPath()).forEach(s -> {
            String[] ss = s.split(",");
            for (int i = 0; i < numFeatures; i++) {
                featureVector[id[0] * numFeatures + i] = ss[i].charAt(0)=='?'?1/512.f:(Double.valueOf(ss[i]));
            }
            String label=ss[ss.length - 1];
            labelVector[id[0] * numLabels + 0] = 0;
            labelVector[id[0] * numLabels + 1] = 0;
            labelVector[id[0] * numLabels + 2] = 0;
            switch (label){
                case "Iris-versicolor":
                    labelVector[id[0] * numLabels + 0]=1;
                    break;
                case "Iris-virginica":
                    labelVector[id[0] * numLabels + 1]=1;
                    break;
                case "Iris-setosa":
                    labelVector[id[0] * numLabels + 2]=1;
                    break;
            }
            id[0]++;
        });

        INDArray x = Nd4j.create(featureVector, new int[]{featureVector.length / numFeatures, numFeatures}, 'c');
        INDArray y = Nd4j.create(labelVector, new int[]{labelVector.length / numLabels, numLabels}, 'c');

        for(int i=0;i<numFeatures;i++) {
            INDArray max = x.getColumn(i).max();
            INDArray min = x.getColumn(i).min();
            for(int j=0;j<numInstances;j++){
                if(x.getColumn(i).getDouble(j)==1/512.f){
                    double median=x.getColumn(i).median(0).getDouble(0);
                    x.getColumn(i).putScalar(j,median);
                }
            }
            double toscale=max.getDouble(0)-min.getDouble(0);
            x.getColumn(i).subi(min);
            x.getColumn(i).divi(toscale);
            x.getColumn(i).muli(2);
            x.getColumn(i).subi(1);
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

        List<DataSet> tdataset = allData.asList();
        datasets=new ArrayList<>();
        int foldsize=numInstances/4;
        if(!train) {
            datasets = tdataset.subList(foldidx*foldsize, foldidx*foldsize+foldsize);
        }
        else{
            for(int i=0;i<numInstances;i++){
                if(i<foldidx*foldsize||i>(foldidx+1)*foldsize){
                    datasets.add(tdataset.get(i));
                }
            }
        }

        iterator = new ListDataSetIterator(datasets,batchSize);
        //iterator.setPreProcessor(org.nd4j.linalg.dataset.api.DataSet::scale);
    }
}
