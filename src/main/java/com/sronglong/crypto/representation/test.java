//package com.isaac.stock.representation;
//
//public class test {
//
//    String csvFile = "D:/p1.csv";
//    BufferedReader br;
//    String line;
//    String cvsSplitBy = ",";
//    int value[] = new int [100];
//    int i=0;
//    int max=0;
//    int min =100;
//        try
//    {
//        br = new BufferedReader(new FileReader(csvFile));
//        while ((line = br.readLine()) != null)
//        {
//            value[i]=Integer.parseInt(line);
//            System.out.println(value[i]);
//            if(value[i]>=max)
//                max = value[i];
//            if(value[i]<min)
//                min = value[i];
//            i++;
//        }
//        br.close();
//        int new_max=Integer.parseInt(jTextField1.getText());
//        int new_min=Integer.parseInt(jTextField2.getText());
//        float a = new_max-new_min;
//        float b = max -min;
//        System.out.println("Value by Min Max Normalization");
//        for(int j =0;j<=i;j++)
//        {
//            float V=((value[j]-min)*a/b)+new_min;
//            System.out.print(V+",");
//        }
//        System.out.println("\nValue by Z-score Normalization");
//        int mean =find_mean(value,i);
//        int var = find_variance(value,mean,i);
//        double std_dev =Math.sqrt(var);
//        for(int j =0;j<=i;j++)
//        {
//            double V1=(value[j]-mean)/std_dev;
//            System.out.print(V1+",");
//        }
//
//
//        System.out.println("\nValue by Decimal Scaling Normalization");
//        double k = Math.pow(10,2);
//        for(int j =0;j<=i;j++)
//        {
//            double V2=value[j]/k;
//            System.out.print(V2+",");
//        }
//    }
//        catch (Exception e)
//    {
//        System.out.println(e);
//    }
//}
//    int find_mean(int value[],int i)
//    {
//        int mean;
//        int sum=0;
//        for(int j=0;j<i;j++)
//        {
//            sum=sum+value[j];
//        }
//        mean=Math.round(sum/i);
//        return mean;
//    }
//    int find_variance(int value[],int mean,int i)
//    {
//        int var;
//        int sum=0;
//        int temp;
//        for(int j=0;j<i;j++)
//        {
//            temp=value[j]-mean;
//            sum= sum + temp*temp;
//        }
//        var=Math.round(sum/i);
//        return var;
//    }
//
//}
