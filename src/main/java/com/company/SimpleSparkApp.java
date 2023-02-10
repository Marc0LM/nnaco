package com.company;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import java.util.ArrayList;
import java.util.List;

public class SimpleSparkApp {
    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("Test spark")
                .setMaster("spark://192.168.0.102:7077")
                .set("spark.blockManager.port", "10025")
                .set("spark.driver.blockManager.port", "10026")
                .set("spark.driver.port", "10027") //make all communication ports static (not necessary if you disabled firewalls, or if your nodes located in local network, otherwise you must open this ports in firewall settings)
                .set("spark.cores.max", "8")
                .set("spark.executor.memory", "512m")
                .set("spark.driver.host", "192.168.0.103")
                .setJars(new String[]{"target/NNACOSP-1.0-SNAPSHOT.jar"});
        //conf.setMaster("local");
        JavaSparkContext sc=new JavaSparkContext(conf);
        //JavaSparkContext sc=new JavaSparkContext("spark://192.168.217.128:7077","NNACOSP");
        int NUM_SAMPLES=1024*1024;
        List<Integer> l = new ArrayList<>(NUM_SAMPLES);
        for (int i = 0; i < NUM_SAMPLES; i++) {
            l.add(i);
        }
        //sc.parallelize(l).filter(i->i>0);
        System.out.println("11111111111111111111111111");
        long count = sc.parallelize(l).filter(new Function<Integer, Boolean>() {
            @Override
            public Boolean call(Integer integer) throws Exception {
                double x = Math.random();
                double y = Math.random();
                return x * x + y * y < 1;
            }
        }).count();
        System.out.println("Pi is roughly " + 4.0 * count / NUM_SAMPLES);
    }
}
