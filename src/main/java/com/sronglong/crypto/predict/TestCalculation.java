package com.sronglong.crypto.predict;

import com.sronglong.crypto.utils.EvaluationMatrix;

import java.lang.reflect.Array;

public class TestCalculation {
    public static void main (String[] args)  {

        double[] actual = {112.3,108.4,148.9,117.4};
        double[] predict =  {124.7,103.7,116.6,78.5};

        System.out.println(EvaluationMatrix.mape(actual,predict));

    }
}
