package com.sronglong.crypto.utils;

import org.apache.commons.math3.analysis.function.Atanh;

public class EvaluationMatrix {
    public static double mseCal(double[] actual, double[] predict){
        int n = actual.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {    double diff = actual[i] - predict[i];
            sum = sum + (diff * diff);
        }
        double mse = sum / n;

        double rmse = Math.sqrt(mse);
       // log.info("rmse : " + rmse);
        return  mse;
    }

    public   static double rmseCal(double mse){
        return   Math.sqrt(mse);
    }


    public   static double maeCal(double[] actual, double[] predict) {
        int n = actual.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum = sum + Math.abs(actual[i] - predict[i]);
        }

        double mae = sum / n;
        return mae;

    }


    public static double mape(double[] actual, double[] predict){
        int n = actual.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum = sum + Math.abs((actual[i] - predict[i])/actual[i]);
        }

        double mape = (sum / n) * 100;
        return mape;
    }


    public static double deTanh(double data, double sdt, double mean) {
      //  res[i] = atanh.value( data[i] / 0.5  - 1) / 0.01 * value_deviation + value_mean;
        return Math.atan( data / 0.5  - 1) / 0.01 * sdt + mean;

    }
}
