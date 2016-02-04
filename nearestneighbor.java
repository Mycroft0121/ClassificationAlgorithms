import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Rain Xuanyu Zhang
 *         zhan2223
 *         4642453
 */
class Main
{
    /**
     * @param args the command line arguments:
     *
     * arg[0] classifier-name:     centroid/regression
     * arg[1] input-file:          file path to 20newsgroups_char.ijv/20newsgroups_word.ijv
     * arg[2] input-rlabel-file:   file path to 20newsgroups.rlabel (didn't use this in the program)
     * arg[3] train-file:          file path to 20newsgroups.train; for ridge reg, it should be 20newsgroups_ridge.train
     * arg[4] test-file:           file path to 20newsgroups.test; for ridge reg, it should be 20newsgroups_ridge.val
     * arg[5] class-file:          file path to 20newsgroups.class
     * arg[6] features-label-file: didn't use this, please provide some value as filler
     * arg[7] feature-representation-option:   tf/binary/sqrt/tfidf/binaryidf/sqrtidf
     * arg[8] output-file:         up to you
     * arg[9] [option]:            you may only provide the file path to the 20newsgroups_ridge.val file
     *
     */
    public static void main(String[] args) throws IOException
    {
        System.out.println("...processing");
        System.out.println("Please be patient, the whole process may take up to four(4) minutes.");

        CentroidClassifier cc = new CentroidClassifier();

        if (args.length >= 9)
        {
            try
            {
                //String ijvFilePath = "/Users/rainzhang/Desktop/project3-files/20newsgroups_word.ijv";
                //String trainSetFilePath = "/Users/rainzhang/Desktop/project3-files/20newsgroups.train";
                //String testSetFilePath = "/Users/rainzhang/Desktop/project3-files/20newsgroups.test";
                //Step4 get average vectors and classify
                //String classFilePath = "/Users/rainzhang/Desktop/project3-files/20newsgroups.class";
                //String resultFilepath = "/Users/rainzhang/Desktop/project3-files/result.csv";
                String classFilePath = args[5];
                Map<String, List<Integer>> ClassLabel = cc.processClassLabel(classFilePath);
                String ijvFilePath = args[1];
                String trainSetFilePath = args[3];
                String testSetFilePath = args[4];
                String resultFilePath = args[8];
                Integer k = Integer.parseInt(args[9]);
                Map<Integer, Map<Integer, Double>> data = new HashMap<>();
                Map<Integer, Map<Integer, Double>> train = new HashMap<>();
                Map<Integer, Map<Integer, Double>> test = new HashMap<>();
                Map<Integer, Double> IDF = new HashMap<>();
                Map<Integer, Map<Integer, Double>> TFIDF = new HashMap<>();

                switch (args[7])
                {
                    case "tf":
                        data = cc.getTF(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //normalize both sets
                        cc.normalizeVector(test);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, test, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    case "binary":
                        data = cc.getBinary(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //normalize both sets
                        cc.normalizeVector(test);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, test, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    case "sqrt":
                        data = cc.getSqrtTF(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //normalize both sets
                        cc.normalizeVector(test);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, test, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    case "tfidf":
                        data = cc.getTF(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //Step1 get IDF from training set
                        IDF = cc.getIDF(train);
                        //Step2 apply the IDF to test set
                        TFIDF = cc.getRepWithIDF(IDF, test);
                        //Step3 normalize both sets
                        cc.normalizeVector(TFIDF);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, TFIDF, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    case "binaryidf":
                        data = cc.getBinary(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //Step1 get IDF from training set
                        IDF = cc.getIDF(train);
                        //Step2 apply the IDF to test set
                        TFIDF = cc.getRepWithIDF(IDF, test);
                        //Step3 normalize both sets
                        cc.normalizeVector(TFIDF);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, TFIDF, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    case "sqrtidf":
                        data = cc.getSqrtTF(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //Step1 get IDF from training set
                        IDF = cc.getIDF(train);
                        //Step2 apply the IDF to test set
                        TFIDF = cc.getRepWithIDF(IDF, test);
                        //Step3 normalize both sets
                        cc.normalizeVector(TFIDF);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, TFIDF, ClassLabel, resultFilePath, classFilePath, k);
                        break;

                    default:
                        data = cc.getTF(ijvFilePath);
                        train = cc.getTrainingSet(data, trainSetFilePath);
                        test = cc.getTestingSet(data, testSetFilePath);
                        //normalize both sets
                        cc.normalizeVector(test);
                        cc.normalizeVector(train);
                        //Step4 centroid based classifier, and output evaluation
                        cc.KNNClassifiy(train, test, ClassLabel, resultFilePath, classFilePath, k);
                        break;
                }
            }
            catch (NullPointerException e)
            {
                e.printStackTrace();
            }
        }
        else
        {
            System.out.println("No sufficient arguments! Make sure you have more than 8 args.");
        }
    }
}

class CentroidClassifier
{
    public void KNNClassifiy (Map<Integer, Map<Integer, Double>> trainingSet,
            Map<Integer, Map<Integer, Double>> testSet,
            Map<String, List<Integer>> classLabel,
            String resultFilePath,
            String Nothing,
            Integer k)
    {
        BufferedWriter bw = null;
        try
        {
            bw = new BufferedWriter(new FileWriter(resultFilePath));
         for (Map.Entry<Integer, Map<Integer,Double>> testSetElmt : testSet.entrySet())
        {
            double currMaxScore = 0.0;
            String assignClass = "";
            for (Map.Entry<String, List<Integer>> currClass : classLabel.entrySet())
            {
                Map<Integer,Map<Integer, Double>> thisClassTrainingElmts  = new HashMap<>(); //+ve
                //Map<Integer,Map<Integer, Double>> otherClassTrainingElmts = new HashMap<>(); //-ve

                for (Map.Entry<Integer, Map<Integer,Double>> trainSetElmt : trainingSet.entrySet())
                {
                    if (currClass.getValue().contains(trainSetElmt.getKey()))
                    {
                        thisClassTrainingElmts.put(trainSetElmt.getKey(), trainSetElmt.getValue());
                    }
                    //else
                    //{
                    //    otherClassTrainingElmts.put(trainSetElmt.getKey(), trainSetElmt.getValue());
                    //}
                }


                double score = 0.0;
                List<DocIDPredScorePair> pair = new ArrayList<>();
                for (Map.Entry<Integer, Map<Integer,Double>> trainingSetElmt : trainingSet.entrySet())
                {
                    pair.add(new DocIDPredScorePair(trainingSetElmt.getKey(), cosineSim(testSetElmt.getValue(), trainingSetElmt.getValue())));

                }

                Collections.sort(pair);
                double p = 0.0;
                double n = 0.0;
                for (int i = 0; i < k; i++)
                {
                    if (thisClassTrainingElmts.get(pair.get(i).DocID) != null)
                    {
                        p += pair.get(i).PredScore;
                    }
                    else
                    {
                        n = pair.get(i).PredScore;
                    }
                }
                score = p - n;

                if (score > currMaxScore)
                {
                    currMaxScore = score;
                    assignClass = currClass.getKey();
                }
            }
            bw.write(testSetElmt.getKey()+","+assignClass);
            bw.newLine();
        }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            try
            {
                if (bw != null)
                    bw.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }




    // step 1. read data file to hashmap
    public Map<Integer, Map<Integer, Double>> getTF (String filepath)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(filepath)))
        {
            String line = br.readLine();
            Map<Integer, Map<Integer, Double>> rawData = new HashMap<>();
            int currentDocID = 1;
            Map<Integer, Double> vector = new HashMap<>();
            while (line != null)
            {
                String[] ijv = line.split(" ");
                int thisDocID = Integer.parseInt(ijv[0]);
                int vectorDimension = Integer.parseInt(ijv[1]);
                double dimensionFreq = Double.parseDouble(ijv[2]);

                if (thisDocID == currentDocID)
                {
                    vector.put(vectorDimension, dimensionFreq);
                }
                else
                {
                    rawData.put(currentDocID, vector);
                    currentDocID = thisDocID;
                    vector = new HashMap<>();
                    vector.put(vectorDimension, dimensionFreq);
                }
                line = br.readLine();
            }
            return rawData;
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public Map<Integer, Map<Integer, Double>> getBinary (String filepath)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(filepath)))
        {
            String line = br.readLine();
            Map<Integer, Map<Integer, Double>> rawData = new HashMap<>();
            int currentDocID = 1;
            Map<Integer, Double> vector = new HashMap<>();
            while (line != null)
            {
                String[] ijv = line.split(" ");
                int thisDocID = Integer.parseInt(ijv[0]);
                int vectorDimension = Integer.parseInt(ijv[1]);

                if (thisDocID == currentDocID)
                {
                    vector.put(vectorDimension, 1.0);
                }
                else
                {
                    rawData.put(currentDocID, vector);
                    currentDocID = thisDocID;
                    vector = new HashMap<>();
                    vector.put(vectorDimension, 1.0);
                }
                line = br.readLine();
            }
            return rawData;
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public Map<Integer, Map<Integer, Double>> getSqrtTF (String filepath)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(filepath)))
        {
            String line = br.readLine();
            Map<Integer, Map<Integer, Double>> rawData = new HashMap<>();
            int currentDocID = 1;
            Map<Integer, Double> vector = new HashMap<>();
            while (line != null)
            {
                String[] ijv = line.split(" ");
                int thisDocID = Integer.parseInt(ijv[0]);
                int vectorDimension = Integer.parseInt(ijv[1]);
                double dimensionFreq = Math.sqrt(Double.parseDouble(ijv[2]));

                if (thisDocID == currentDocID)
                {
                    vector.put(vectorDimension, dimensionFreq);
                }
                else
                {
                    rawData.put(currentDocID, vector);
                    currentDocID = thisDocID;
                    vector = new HashMap<>();
                    vector.put(vectorDimension, dimensionFreq);
                }
                line = br.readLine();
            }
            return rawData;
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public Map<Integer, Map<Integer, Double>> getTrainingSet (Map<Integer, Map<Integer, Double>> rawData, String filepath)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(filepath)))
        {
            String line = br.readLine();
            int thisDocID;
            Map<Integer, Map<Integer, Double>> trainingSet = new HashMap<>();
            while (line != null)
            {
                //System.out.println(line);
                thisDocID = Integer.parseInt(line);
                if (rawData.get(thisDocID) != null)
                {
                    trainingSet.put(thisDocID, rawData.get(thisDocID));
                }
                line = br.readLine();
            }
            return trainingSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public Map<Integer, Map<Integer, Double>> getTestingSet (Map<Integer, Map<Integer, Double>> rawData, String filepath)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(filepath)))
        {
            String line = br.readLine();
            int thisDocID;
            Map<Integer, Map<Integer, Double>> trainingSet = new HashMap<>();
            while (line != null)
            {
                //System.out.println(line);
                thisDocID = Integer.parseInt(line);
                if (rawData.get(thisDocID) != null)
                {
                    trainingSet.put(thisDocID, rawData.get(thisDocID));
                }
                line = br.readLine();
            }
            return trainingSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public Map<Integer, Double> getIDF (Map<Integer, Map<Integer, Double>> trainingSet)
    {
        Map<Integer, Double> IDF = new HashMap<>();
        for (Map.Entry<Integer, Map<Integer, Double>> entry : trainingSet.entrySet())
        {
            Map<Integer, Double> content = entry.getValue();
            for (Map.Entry<Integer, Double> v : content.entrySet())
            {
                if (IDF.get(v.getKey()) == null)
                {
                    IDF.put(v.getKey(), 1.0);
                }
                else
                {
                    double incCounter = IDF.get(v.getKey()) + 1.0;
                    IDF.replace(v.getKey(), incCounter);
                }
            }
        }

        for (Map.Entry<Integer, Double> entry : IDF.entrySet())
        {
            double count = entry.getValue();
            entry.setValue((Math.log(3367.0/count)) / (Math.log(2.0)));
        }
        return IDF;
    }

    public Map<Integer, Map<Integer, Double>> getRepWithIDF(Map<Integer, Double> IDF,
            Map<Integer, Map<Integer, Double>> data)
    {
        Map<Integer, Map<Integer, Double>> result = data;
        for (Map.Entry<Integer, Map<Integer, Double>> entry : result.entrySet())
        {
            Map<Integer, Double> content = entry.getValue();
            for (Map.Entry<Integer, Double> v : content.entrySet())
            {
                if (IDF.get(v.getKey()) != null)
                {
                    double IDFVal = IDF.get(v.getKey());
                    double origVal = v.getValue();
                    v.setValue(IDFVal * origVal);
                }
                else
                {
                    v.setValue(1.0);
                }
            }
        }
        return result;
    }

    public void normalizeVector(Map<Integer, Map<Integer, Double>> data)
    {
        for (Map.Entry<Integer, Map<Integer, Double>> entry : data.entrySet())
        {
            Map<Integer, Double> content = entry.getValue();
            double length = 0.0;
            for (Map.Entry<Integer, Double> v : content.entrySet())
            {
                double temp = v.getValue();
                length += temp*temp;
            }
            length = Math.sqrt(length);
            for (Map.Entry<Integer, Double> v : content.entrySet())
            {
                double temp = v.getValue();
                v.setValue(temp/length);
            }
            entry.setValue(content);
        }
    }

    public Map<String, List<Integer>> processClassLabel (String classFilepath)
    {
        Map<String, List<Integer>> result = new HashMap<>();

        try(BufferedReader br = new BufferedReader(new FileReader(classFilepath)))
        {
            String line = br.readLine();
            String currentClass = "alt.atheism";
            result.put(currentClass, new ArrayList<>());
            while (line != null)
            {
                String[] classLabel = line.split(" ");

                if (currentClass.equals(classLabel[1]))
                {
                    result.get(classLabel[1]).add(Integer.parseInt(classLabel[0]));
                    //System.out.println(classLabel[1] + " temp list added " + classLabel[0] + "current list length = " + tempList.size());
                }
                else
                {
                    currentClass = classLabel[1];
                    result.put(currentClass, new ArrayList<>());
                    result.get(currentClass).add(Integer.parseInt(classLabel[0]));
                }
                line = br.readLine();
            }

        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        //System.out.println(result.size());
        return result;
    }

    public Map<Integer, String> getClassLabel (String classFilePath)
    {
        Map<Integer, String> result = new HashMap<>();
        try(BufferedReader br = new BufferedReader(new FileReader(classFilePath)))
        {
            String line = br.readLine();

            while (line != null)
            {
                String[] classLabel = line.split(" ");
                result.put(Integer.parseInt(classLabel[0]), classLabel[1]);
                line = br.readLine();
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        //System.out.println(result.size());
        return result;
    }

    public Map<Integer, Double> combineVector(Map<Integer, Double> v1, Map<Integer, Double> v2)
    {
        for (Map.Entry<Integer, Double> entry : v2.entrySet())
        {
            if (v1.get(entry.getKey()) == null)
            {
                v1.put(entry.getKey(), entry.getValue());
            }
            else
            {
                double temp = entry.getValue() + v1.get(entry.getKey());
                v1.replace(entry.getKey(), temp);
            }
        }
        return v1;
    }

    public Map<Integer, Double> devideVector(Map<Integer, Double> vector, Double div)
    {
        for (Map.Entry<Integer, Double> entry : vector.entrySet())
        {
            double temp = entry.getValue() / div;
            vector.replace(entry.getKey(), temp);
        }
        return vector;
    }

    public double cosineSim (Map<Integer, Double> v1, Map<Integer, Double> v2)
    {
        double upper = 0.0;
        double lower = 0.0;

        for (Map.Entry<Integer, Double> v1entry : v1.entrySet())
        {
            if (v2.get(v1entry.getKey()) != null)
            {
                upper += v1entry.getValue() * v2.get(v1entry.getKey());
            }
        }

        double temp1 = 0.0;
        for (Double x : v1.values())
        {
            temp1 += x*x;
        }

        double temp2 = 0.0;
        for (Double y : v2.values())
        {
            temp2 += y*y;
        }

        lower = Math.sqrt(temp1) * Math.sqrt(temp2);

        return upper/lower;
    }

    public void centroidClassifier (Map<Integer, Map<Integer, Double>> trainingSet,
            Map<Integer, Map<Integer, Double>> testSet,
            Map<String, List<Integer>> classLabel,
            String resultFilePath,
            String classFilePath)
    {
        List<String> classNameSeq = new ArrayList<>();
        List<Map<Integer, Double>> EachClassAvg = new ArrayList<>();
        List<Map<Integer, Double>> OtherClassAvg = new ArrayList<>();

        for (Map.Entry<String, List<Integer>> classLabelElmt : classLabel.entrySet())
        {
            Map<Integer, Double> tempPve = new HashMap<>();
            Map<Integer, Double> tempNve = new HashMap<>();
            double counterP = 0.0;
            double counterN = 0.0;
            classNameSeq.add(classLabelElmt.getKey());
            for (Map.Entry<Integer, Map<Integer,Double>> trainSetElmt : trainingSet.entrySet())
            {
                if (classLabelElmt.getValue().contains(trainSetElmt.getKey()))
                {
                    Map<Integer, Double> merge = combineVector(tempPve, trainSetElmt.getValue());
                    counterP++;
                }
                else
                {
                    Map<Integer, Double> merge = combineVector(tempNve, trainSetElmt.getValue());
                    counterN++;
                }
            }
            EachClassAvg.add(devideVector(tempPve, counterP));
            OtherClassAvg.add(devideVector(tempNve, counterN));
        }



        BufferedWriter bw = null;
        try
        {
            bw = new BufferedWriter(new FileWriter(resultFilePath));
            for (Map.Entry<Integer, Map<Integer, Double>> testEntry : testSet.entrySet())
            {

                double currentMaxDiff = -10.0;
                String currentAssignedClass = "";
                for (int i = 0; i < 20; i++)
                {
                    double a = cosineSim(EachClassAvg.get(i), testEntry.getValue());
                    double b = cosineSim(OtherClassAvg.get(i), testEntry.getValue());


                    if (a - b > currentMaxDiff)
                    {
                        currentMaxDiff = a - b;
                        if (a > b)
                        {
                            currentAssignedClass = classNameSeq.get(i);
                        }
                        else
                        {
                            currentAssignedClass = "NOT_" + classNameSeq.get(i);
                        }
                        currentAssignedClass = classNameSeq.get(i);
                    }
                }
                bw.write(testEntry.getKey() + "," + currentAssignedClass);
                bw.newLine();
                //System.out.println(testEntry.getKey() + "   " + currentAssignedClass);
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            try
            {
                if (bw != null)
                    bw.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }

        System.out.println("Classification finished, please check output file for assignment results.");
        System.out.println("Now evaluating 20 models...");

        evaluate20Models(classNameSeq, EachClassAvg, OtherClassAvg, testSet, classFilePath);
    }

    public void evaluate20Models(List<String> classNameSeq,
            List<Map<Integer, Double>> EachClassAvgVector,
            List<Map<Integer, Double>> OtherClassAvgVector,
            Map<Integer, Map<Integer, Double>> testSet,
            String classFilePath)
    {
        String bestModel = "";
        double bestMaxF1 = 0.0;

        Map<Integer, String> refSet = getClassLabel(classFilePath);

        for (int i = 0; i < 20; i++)
        {

            List<DocIDPredScorePair> target = new ArrayList<>();
            for (Map.Entry<Integer, Map<Integer, Double>> testSetEntry : testSet.entrySet())
            {
                double a = cosineSim(EachClassAvgVector.get(i), testSetEntry.getValue());
                double b = cosineSim(OtherClassAvgVector.get(i), testSetEntry.getValue());
                double predScore = a - b;
                DocIDPredScorePair temp = new DocIDPredScorePair(testSetEntry.getKey(), predScore);
                target.add(temp);
            }
            Collections.sort(target);
            //TODO : 有了排好序的链表target,现在需要求每个空隙中间的F1值,并找出最大的F1值(不断更新bestModel)
            //在过程中,打印"In model " + classNameSeq.get(i) + "/NOT_" + classNameSeq.get(i) +
            //           " the threshold/max F1(+ve) value is " + maxF1;

            double localMaxF1 = 0.0;

            for (int n = 0; n < target.size(); n++)
            {
                double tempF1 = calculateF1(n, target, classNameSeq.get(i), refSet);
                if (tempF1 > localMaxF1)
                {
                    localMaxF1 = tempF1;
                }
            }
            System.out.println("In model " + classNameSeq.get(i) + " vs. theRest, the max F1(+ve) value is " + localMaxF1);

            if (localMaxF1 > bestMaxF1)
            {
                bestMaxF1 = localMaxF1;
                bestModel = classNameSeq.get(i);
            }
        }
        System.out.println("------- The best model is " + bestModel + " with max F1(+ve) value " + bestMaxF1);
    }

    public double calculateF1(Integer n, List<DocIDPredScorePair> list,
            String currentClassName, Map<Integer, String> refSet)
    {
        double TP = 0.0;
        double FP = 0.0;
        double FN = 0.0;

        for (int i = 0; i < n; i++)
        {
            if (refSet.get(list.get(i).DocID).equals(currentClassName))
            {
                TP++;
            }
            else
            {
                FP++;
            }
        }

        for (int j = n+1; j < list.size(); j++)
        {
            if (refSet.get(list.get(j).DocID).equals(currentClassName))
            {
                FN++;
            }
        }

        double prec = TP/(TP+FP);
        double rec  = TP/(TP+FN);
        double result = (2*prec*rec)/(prec+rec);
        return result;
    }
}

class DocIDPredScorePair implements Comparable<DocIDPredScorePair>
{
    int DocID;
    double PredScore;

    public DocIDPredScorePair(int i, double d)
    {
        DocID = i;
        PredScore = d;
    }

    @Override
    public int compareTo (DocIDPredScorePair o)
    {
        double compareScore = o.PredScore;
        if (this.PredScore > compareScore)
        {
            return -1;
        }
        else if (this.PredScore == compareScore)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    @Override
    public String toString()
    {
        return (Integer.toString(DocID) + " " + Double.toString(PredScore));
    }
}
