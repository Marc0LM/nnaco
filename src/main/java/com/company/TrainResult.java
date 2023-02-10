package com.company;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.LinkedList;
import java.util.List;

class TrainResult {
    List<Pair<Integer, Double>> scores = new LinkedList<>();
    int cursor;

    public TrainResult(int numEpoch) {
    }

    public void addScore(double tacc, int tbatch) {
        scores.add(new MutablePair<>(tbatch, tacc));
    }
}
