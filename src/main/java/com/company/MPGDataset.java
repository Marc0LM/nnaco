package com.company;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;


import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

public class MPGDataset {

    public static final int numFeatures=9;
    public static final int numLabels=1;
    public static final int numInstances=398;

    public List<DataSet> datasets;
    public ListDataSetIterator iterator;
    public MPGDataset(int batchSize, int seed,int foldidx,Boolean train) throws IOException {
        double[] featureVector = new double[numFeatures * numInstances];
        double[] labelVector = new double[numLabels * numInstances];
        int[] id = new int[]{0};
        Files.lines(Paths.get("mpg_data")).forEach(s->{
            String[] values=s.split(" ");
            List<String> vs= new ArrayList<>(Arrays.asList(values));
            vs.removeIf(v-> v.length()==0);
            try {
                labelVector[id[0] * numLabels] = Double.valueOf(vs.get(0));
                //featureVector[id[0] * numFeatures + j]=values[j + 1].charAt(0)=='?'?1/512.f:Double.valueOf(values[j + 1]);
                featureVector[id[0] * numFeatures + 0] = vs.get(1).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(1));
                featureVector[id[0] * numFeatures + 1] = vs.get(2).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(2));
                featureVector[id[0] * numFeatures + 2] = vs.get(3).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(3));
                featureVector[id[0] * numFeatures + 3] = vs.get(4).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(4));
                featureVector[id[0] * numFeatures + 4] = vs.get(5).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(5));
                featureVector[id[0] * numFeatures + 5] = vs.get(6).charAt(0) == '?' ? 1 / 512.f : Double.valueOf(vs.get(6));
                switch (Integer.valueOf(vs.get(7).substring(0, 1))) {
                    case 1:
                        featureVector[id[0] * numFeatures + 6] = 1;
                        break;
                    case 2:
                        featureVector[id[0] * numFeatures + 7] = 1;
                        break;
                    case 3:
                        featureVector[id[0] * numFeatures + 8] = 1;
                        break;
                }
                id[0]++;
            }
            catch (Exception e){
                e.printStackTrace();
            }
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


        iterator = new ListDataSetIterator(datasets, batchSize);

        //iterator.setPreProcessor(org.nd4j.linalg.dataset.api.DataSet::normalizeZeroMeanZeroUnitVariance);
    }
}
