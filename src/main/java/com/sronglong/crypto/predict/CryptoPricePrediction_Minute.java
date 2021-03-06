package com.sronglong.crypto.predict;
//mvn compile exec:java -Dexec.mainClass="com.isaac.stock.predict.StockPricePrediction"
//

import com.sronglong.crypto.model.RecurrentNets;
import com.sronglong.crypto.representation.NormalizeType;
import com.sronglong.crypto.representation.PriceCategory;
import com.sronglong.crypto.representation.StockDataSetIteratorNew;
import com.sronglong.crypto.utils.CsvWriterExamples;
import com.sronglong.crypto.utils.EvaluationMatrix;
import com.sronglong.crypto.utils.Helpers;
import com.sronglong.crypto.utils.PlotUtil;
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
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.NoSuchElementException;

//import com.isaac.stock.representation.*;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class CryptoPricePrediction_Minute {

    private static final Logger log = LoggerFactory.getLogger(CryptoPricePrediction_Minute.class);

    private static int exampleLength = 30; // time series length, assume 22 working days per month

    private static StockDataSetIteratorNew iterator;
    private static String CSV_NAME = "MINUTE";

    public static void main(String[] args) throws IOException {
        String fileTrain = new ClassPathResource("minunte_btc_training.csv").getFile().getAbsolutePath();
        String fileTest = new ClassPathResource("minute_btc_test.csv").getFile().getAbsolutePath();

        int batchSize = 1000; // mini-batch size
        double splitRatio = 0.8; // 90% for training, 10% for testing
        int epochs = 100; // training epochs
        NormalizeType normalizeType = NormalizeType.TANH_EST;
        int type = 0;

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
//         iterator = new CryptoDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);

         iterator = new StockDataSetIteratorNew(fileTrain, batchSize, exampleLength, splitRatio, category,normalizeType);

     //   StockDataSetIteratorNew testiterator = new StockDataSetIteratorNew(fileTest, batchSize, exampleLength, 0, category,normalizeType);

//         StockDataSetIteratorNew  training = iterator;
//        training.spliteTrainandValidate(0.8,true);
//
//        StockDataSetIteratorNew  validate = iterator;
//        training.spliteTrainandValidate(0.8,false);


        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
//        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        log.info("Training...");
        net.setListeners(new ScoreIterationListener(100));

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        net.setListeners(new StatsListener(statsStorage));


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
        File locationToSave = new File("src/main/resources/Minute_LSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");

        predictPriceOneAhead(net, test, category,normalizeType);

        log.info("Done...");

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        System.out.println(eval.stats());

    }


    /**
     * Predict one feature of a stock one-day ahead
     */
    private static void predictPriceOneAhead(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, PriceCategory category, NormalizeType normalizeType) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        double[] predictsnormalized = new double[testData.size()];
        double[] actualsnormalizerd = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            INDArray ma1x = net.rnnTimeStep(testData.get(i).getKey());

            predicts[i] = denormalize(net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1),normalizeType);
//            predicts[i] = EvaluationMatrix.deTanh(net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1),value_deviation,value_mean);
            actuals[i] = testData.get(i).getValue().getDouble(0);

            predictsnormalized[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1);
            actualsnormalizerd[i] = testData.get(i).getValue().getDouble(1);

        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predictsnormalized[i] + "," + actualsnormalizerd[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));

        log.info(writeFile(actuals,predicts,predictsnormalized,actualsnormalizerd,CSV_NAME + normalizeType.toString()));



//        MultiLayerNetwork

        //evaluate the model on the test set
//        RegressionEvaluation eval =  new RegressionEvaluation(0);
//        INDArray predict = Nd4j.create(predicts);
//        INDArray acuatl = Nd4j.create(actuals);
//        eval.eval(acuatl,predict);
////        Evaluation eval = net.evaluate(testData);
//        log.info(eval.stats());

//        double[] actual, pred
        double mse = EvaluationMatrix.mseCal(actuals, predicts);
        log.info("mse : " + mse);
        log.info("rmse : " + EvaluationMatrix.rmseCal(mse));
        log.info("mae : " + EvaluationMatrix.maeCal(actuals, predicts));

    }

    public static String writeFile(double[] predicts, double[] actuals,double[] predictsnor, double[] actualsnor,String name){
        Path path = null;
        try {
            path = Helpers.fileOutOnePath(name);
        } catch (Exception ex) {
            Helpers.err(ex);
        }

        return CsvWriterExamples.csvWriterAll(CsvWriterExamples.toStringList(actuals,predicts,actualsnor, predictsnor,name),path);
    }

    public static double denormalize(double value, NormalizeType normalizeType){
//        DECIMAL_SCALING,Z_SCORE,MEDIAN_NOR,SIGMOID_NOR, TANH_EST;
        switch (normalizeType){
            case MINMAX: return iterator.denormalMinMAx(value,1);
            case DECIMAL_SCALING: return iterator.dedecimalScalingNormalization(value,1);
            case Z_SCORE: return iterator.dezScore(value,1);
            case MEDIAN_NOR: return iterator.demedianNormalization(value,1);
            case SIGMOID_NOR: return iterator.desigmoidNormalization(value,1);
            case TANH_EST: return iterator.detanhestimators(value,1);
            default: throw new NoSuchElementException();
        }

    }


}