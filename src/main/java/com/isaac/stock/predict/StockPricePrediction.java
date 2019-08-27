package com.isaac.stock.predict;
//mvn compile exec:java -Dexec.mainClass="com.isaac.stock.predict.StockPricePrediction"
//
import com.isaac.stock.model.RecurrentNets;
import com.isaac.stock.representation.CryptoBTCDataSetIterator;
import com.isaac.stock.representation.CryptoDataSetIterator;
import com.isaac.stock.representation.PriceCategory;
import com.isaac.stock.representation.StockDataSetIterator;
import com.isaac.stock.utils.EvaluationMatrix;
import com.isaac.stock.utils.PlotUtil;
import javafx.util.Pair;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.eval.Evaluation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class StockPricePrediction {

    private static final Logger log = LoggerFactory.getLogger(StockPricePrediction.class);

    private static int exampleLength = 30; // time series length, assume 22 working days per month
    private static CryptoDataSetIterator iterator;
    public static void main (String[] args) throws IOException {
        String file = new ClassPathResource("gemini_BTCUSD_2019_1min-2.csv").getFile().getAbsolutePath();
        String symbol = "GOOG"; // stock name
        int batchSize = 64; // mini-batch size
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 1; // training epochs

        int type = 1;

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
         iterator = new CryptoDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
//        CryptoBTCDataSetIterator iterator = new CryptoBTCDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

//        //Initialize the user interface backend
//        UIServer uiServer = UIServer.getInstance();
//
//        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
//
//        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//        uiServer.attach(statsStorage);
//
//        //Then add the StatsListener to collect this information from the network, as it trains
//        net.setListeners(new StatsListener(statsStorage));
        net.setListeners(new ScoreIterationListener(100));

        log.info("Training...");
        long timeX = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            long time1 = System.currentTimeMillis();

            while (iterator.hasNext()) net.fit(iterator.next()); // fit model using mini-batch data
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, time: {} ***", i, (time2 - time1));
        }

        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");
        if (type == 1) {
            //predict with tanh :
            double mean = iterator.getMinNum(category);
            double sdv = iterator.getMaxNum(category);
            predictPriceWithTanh(net,test,mean,sdv,category);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net, test, max, min, category);
        }




        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            INDArray ma1x = net.rnnTimeStep(testData.get(i).getKey());
            log.info("in");
            System.out.println(ma1x);
            log.info("out");
            System.out.println(testData.get(i).getKey());

            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));

//        MultiLayerNetwork

        //evaluate the model on the test set
//        RegressionEvaluation eval =  new RegressionEvaluation(0);
//        INDArray predict = Nd4j.create(predicts);
//        INDArray acuatl = Nd4j.create(actuals);
//        eval.eval(acuatl,predict);
//        Evaluation eval = net.evaluate(testData);
//        log.info(eval.stats());

//        double[] actual, pred
        double mse = EvaluationMatrix.mseCal(actuals,predicts);
        log.info("mse : " + mse);
        log.info("rmse : " + EvaluationMatrix.rmseCal(mse) );
        log.info("mae : " + EvaluationMatrix.maeCal(actuals,predicts));

    }


    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceWithTanh (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double mean, double sdv, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            INDArray ma1x = net.rnnTimeStep(testData.get(i).getKey());
//            log.info("in");
//            System.out.println(ma1x);
//            log.info("out");
//            System.out.println(testData.get(i).getKey());
            print(ma1x,"predict");
            print(testData.get(i).getKey(),"actual");
            predicts[i] = EvaluationMatrix.deTanh(net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1),sdv,mean );
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        System.out.println(eval.stats());


//        MultiLayerNetwork

        //evaluate the model on the test set
//        RegressionEvaluation eval =  new RegressionEvaluation(0);
//        INDArray predict = Nd4j.create(predicts);
//        INDArray acuatl = Nd4j.create(actuals);
//        eval.eval(acuatl,predict);
//        Evaluation eval = net.evaluate(testData);
//        log.info(eval.stats());

//        double[] actual, pred
        double mse = EvaluationMatrix.mseCal(actuals,predicts);
        log.info("mse : " + mse);
        log.info("rmse : " + EvaluationMatrix.rmseCal(mse) );
        log.info("mae : " + EvaluationMatrix.maeCal(actuals,predicts));

    }


    private static void print( INDArray myArray, String type){
        //Next, print some basic information about the array:
        System.out.println("Basic INDArray information: " + type);
//        System.out.println("Num. Rows:          " + myArray.rows());
//        System.out.println("Num. Columns:       " + myArray.columns());
//        System.out.println("Num. Dimensions:    " + myArray.rank());                    //2 dimensions -> rank 2
//        System.out.println("Shape:              " + Arrays.toString(myArray.shape()));  //[3,5] -> 3 rows, 5 columns
//        System.out.println("Length:             " + myArray.length());                  // 3 rows * 5 columns = 15 total elements
      for (int i= 0; i<myArray.rows(); i++){
          System.out.print(myArray.getDouble(i)+" , ");
      }


    }


    private static void predictPriceMultiple (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min) {
        // TODO
    }

    /** Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead */
    private static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\tActual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "\t" + actuals[i]);
        log.info("Plot...");
        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0: name = "Stock OPEN Price"; break;
                case 1: name = "Stock CLOSE Price"; break;
                case 2: name = "Stock LOW Price"; break;
                case 3: name = "Stock HIGH Price"; break;
                case 4: name = "Stock VOLUME Amount"; break;
                default: throw new NoSuchElementException();
            }
            PlotUtil.plot(pred, actu, name);
        }
    }

}
