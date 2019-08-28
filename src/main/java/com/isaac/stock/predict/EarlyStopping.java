package com.isaac.stock.predict;
//mvn compile exec:java -Dexec.mainClass="com.isaac.stock.predict.StockPricePrediction"
//

import com.isaac.stock.model.RecurrentNets;
import com.isaac.stock.representation.*;
import com.isaac.stock.utils.EvaluationMatrix;
import com.isaac.stock.utils.PlotUtil;
import javafx.util.Pair;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
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
import java.util.*;
import java.util.concurrent.TimeUnit;

//import org.deeplearning4j.parallelism.ParallelWrapper;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class EarlyStopping {

    private static final Logger log = LoggerFactory.getLogger(EarlyStopping.class);

    private static int exampleLength = 30; // time series length, assume 22 working days per month
    private static CryptoDataSetIterator iterator;
    public static void main (String[] args) throws IOException {
        String fileTrain = new ClassPathResource("BTC_daily__training.csv").getFile().getAbsolutePath();
        String fileTest = new ClassPathResource("BTC_daily_testdata.csv").getFile().getAbsolutePath();

        int batchSize = 64; // mini-batch size
        double splitRatio = 1; // 90% for training, 10% for testing
        int epochs = 1; // training epochs
        NormalizeType normalizeType = NormalizeType.MINMAX;
        int type = 0;

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
//         iterator = new CryptoDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);

//        StockDataSetIteratorNew myTrainData = new StockDataSetIteratorNew(fileTrain, batchSize, exampleLength, splitRatio, category,normalizeType);
//        StockDataSetIteratorNew myTestData = new StockDataSetIteratorNew(fileTest, batchSize, exampleLength, splitRatio, category,normalizeType);

        StockDataSetIteratorNew iterator = new StockDataSetIteratorNew(fileTrain, batchSize, exampleLength, splitRatio, category,normalizeType);

        StockDataSetIteratorNew  myTrainData = iterator;
        myTrainData.spliteTrainandValidate(0.8,true);

        StockDataSetIteratorNew  myTestData = iterator;
        myTestData.spliteTrainandValidate(0.8,false);


        log.info("Load test dataset...");
     //   List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(myTrainData.inputColumns(), myTrainData.totalOutcomes());

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        System.out.println(exampleDirectory);
        File dirFile = new File(exampleDirectory); //We have to create the temp directory or the sample will fail.
        dirFile.mkdir(); // If mkdir fails, it is probably because the directory already exists. Which is fine.
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100), new ScoreImprovementEpochTerminationCondition(8)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
                .scoreCalculator(new DataSetLossCalculator(myTestData, true))     //Calculate test set score
                .modelSaver(saver)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,myTrainData);

        //Conduct early stopping training:
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }

        //Get the best model:
        MultiLayerNetwork bestModel = result.getBestModel();
        RegressionEvaluation eval = bestModel.evaluateRegression(myTestData);
        System.out.println(eval.stats());

        log.info("Done...");
    }


}
