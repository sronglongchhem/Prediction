package com.sronglong.crypto.predict;

import com.sronglong.crypto.representation.NormalizeData;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

public class CreateNormalizedData {
    public static void main (String[] args) throws IOException {
        String fileTrain = new ClassPathResource("BTC_daily__training.csv").getFile().getAbsolutePath();
        String fileTrain1 = new ClassPathResource("ETH_daily__training.csv").getFile().getAbsolutePath();
        NormalizeData normalized = new NormalizeData(fileTrain,"BTC");
        NormalizeData normalizedETH = new NormalizeData(fileTrain1,"ETH");
    }

}
