package com.isaac.stock.utils;

public class EvaluationMatrix {
    public static double mseCal(double[] actual, double[] predict){
        int n = actual.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {    double diff = actual[i] - predict[i];
            sum = sum + diff * diff;
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
}
