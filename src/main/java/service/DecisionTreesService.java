package service;

import model.*;
import repository.FileRepo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.Collectors;

public class DecisionTreesService {

    private int numberOfPossibleOutputs;
    private FileRepo repo;

    public DecisionTreesService(int numberOfPossibleOutputs, FileRepo repo) {
        this.numberOfPossibleOutputs = numberOfPossibleOutputs;
        this.repo = repo;
    }

    public Double oneAttributeEntropy(Matrix<Double> results){
        Double entropy=0.0;
        ArrayList<Double> proportions = new ArrayList<>();
        for(int i=0;i<numberOfPossibleOutputs;i++){
            proportions.add(0.0);
        }
        for(int i=0;i<results.getRows();i++){
            Integer index = (int) Math.round(results.get(i,0)/0.5);
            proportions.set(index,proportions.get(index)+1);
        }
        for(int i=0;i<numberOfPossibleOutputs;i++){
            if(proportions.get(i)!=0) {
                proportions.set(i, proportions.get(i) / results.getRows());
                entropy += -(proportions.get(i) * (Math.log(proportions.get(i)) / Math.log(2)));
            }
        }
        return entropy;
    }



    public Pair<Double,Double> bestSplitForAttribute(Double generalEntropy,Matrix<Double> data,Matrix<Double> results,Integer attribute){
        Pair<Double,Double> bestSplit=null;
        ArrayList<Double> values = new ArrayList<>();
        for(int i=0;i<data.getRows();i++){
            values.add(data.get(i,attribute));
        }
        values.sort((v1,v2)->{
            if(v1>v2){
                return 1;
        }
            else if(v1<v2){
                return -1;
            }
            else return 0;
        });
        for(int i=0;i<values.size()-1;i++){
            Double threshold = (values.get(i)+values.get(i+1))/2;
            Integer numberOfValuesLowerThanThreshold=0;
            Integer numberOfValuesHigherThanThreshold=0;
            Matrix<Double> matrixOfValuesLowerThanThreshold = new Matrix<>(data.getRows(),results.getColumns(),0.0);
            Matrix<Double> matrixOfValuesHigherThanThreshold = new Matrix<>(data.getRows(),results.getColumns(),0.0);
            for(int j=0;j<data.getRows();j++){
                if(data.get(j,attribute)<=threshold){
                    matrixOfValuesLowerThanThreshold.setRow(results.getRow(j),numberOfValuesLowerThanThreshold);
                    numberOfValuesLowerThanThreshold++;
                }
                else{
                    matrixOfValuesHigherThanThreshold.setRow(results.getRow(j),numberOfValuesHigherThanThreshold);
                    numberOfValuesHigherThanThreshold++;
                }
            }
            matrixOfValuesLowerThanThreshold.setRows(numberOfValuesLowerThanThreshold);
            matrixOfValuesHigherThanThreshold.setRows(numberOfValuesHigherThanThreshold);
            Double informationGain = generalEntropy - (((double)numberOfValuesLowerThanThreshold/data.getRows())*oneAttributeEntropy(matrixOfValuesLowerThanThreshold)
                    +((double)numberOfValuesHigherThanThreshold/data.getRows())*oneAttributeEntropy(matrixOfValuesHigherThanThreshold));
            if(bestSplit==null){
                bestSplit = new Pair<>(threshold,informationGain);
            }
            else{
                if(bestSplit.getSecond()<informationGain){
                    bestSplit = new Pair<>(threshold,informationGain);
                }
            }

        }
        return bestSplit;
    }

    public Pair<Integer,Double> getAttributeWithBestInformationGainAndBestSplit(Matrix<Double> data,Matrix<Double> results){
        Double generalEntropy = oneAttributeEntropy(results);
        Pair<Integer,Double> bestAttributeAndSplit = null;
        Double bestGain=-1.0;
        for(int i=0;i<data.getColumns();i++){
            Pair<Double,Double> bestSplitAndGainForAttribute = bestSplitForAttribute(generalEntropy,data,results,i);
            if(bestAttributeAndSplit==null){
                bestAttributeAndSplit = new Pair<>(i, bestSplitAndGainForAttribute.getFirst());
                bestGain = bestSplitAndGainForAttribute.getSecond();
            }
            else{
                if(bestSplitAndGainForAttribute.getSecond()>bestGain){
                    bestAttributeAndSplit = new Pair<>(i, bestSplitAndGainForAttribute.getFirst());
                    bestGain = bestSplitAndGainForAttribute.getSecond();
                }
            }
        }
        return bestAttributeAndSplit;
    }

    public boolean checkIfResultAreFromSameClass(Matrix<Double> results){
        Double firstValue = results.get(0,0);
        for(int i=0;i<results.getRows();i++){
            if(!results.get(i,0).equals(firstValue)){
                return false;
            }
        }
        return true;
    }

    public Double getMajorityClass(Matrix<Double> results){
        ArrayList<Integer> frequencyVector =  new ArrayList<>();
        for(int i=0;i<numberOfPossibleOutputs;i++){
            frequencyVector.add(0);
        }
        for(int i=0;i<results.getRows();i++){
            Integer index = (int) Math.round(results.get(i,0)/0.5);
            frequencyVector.set(index,frequencyVector.get(index)+1);
        }
        Double max = 0.0;
        Integer maxFrequency = -1;
        for(int i=0;i<numberOfPossibleOutputs;i++){
            if(frequencyVector.get(i)>maxFrequency){
                maxFrequency = frequencyVector.get(i);
                max = 0.5*i;
            }
        }
        return max;
    }

    private Node<Pair<NodeType,Double>> generateTree(Matrix<Double> data,Matrix<Double> results){
        Node<Pair<NodeType,Double>> newNode;
        if(checkIfResultAreFromSameClass(results)){
            newNode = new Node<>(new Pair<>(NodeType.Leaf,results.get(0,0)));
            return newNode;
        }
        else{
            if(data.getColumns()==0){
                Double majorityClass = getMajorityClass(results);
                newNode = new Node<>(new Pair<>(NodeType.Leaf,majorityClass));
                return newNode;
            }
            else{
                Pair<Integer,Double> separationAttribute = getAttributeWithBestInformationGainAndBestSplit(data,results);
                newNode = new Node<>(new Pair<>(NodeType.SeparationAttribute,(double)separationAttribute.getFirst()));
                newNode.setThreshold(separationAttribute.getSecond());
                Matrix<Double> dataMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold = new Matrix<>(data.getRows(),data.getColumns()-1,0.0);
                Matrix<Double> dataMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold = new Matrix<>(data.getRows(),data.getColumns()-1,0.0);
                Matrix<Double> resultMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold = new Matrix<>(data.getRows(),results.getColumns(),0.0);
                Matrix<Double> resultMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold = new Matrix<>(data.getRows(),results.getColumns(),0.0);
                Integer dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter = 0;
                Integer dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter = 0;
                Integer offset;
                for(int i=0;i<data.getRows();i++){
                    offset = 0;
                    for(int j=0;j<data.getColumns();j++){
                        if(j==separationAttribute.getFirst()){
                            offset = -1;
                        }
                        else{
                            if(data.get(i,separationAttribute.getFirst())<=separationAttribute.getSecond()){
                                dataMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold.set(data.get(i,j),dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter,j+offset);
                            }
                            else{
                                dataMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold.set(data.get(i,j),dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter,j+offset);
                            }
                        }
                    }
                    if(data.get(i,separationAttribute.getFirst())<=separationAttribute.getSecond()){
                        resultMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold.set(results.get(i,0),dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter,0);
                        dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter++;
                    }
                    else{
                        resultMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold.set(results.get(i,0),dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter,0);
                        dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter++;
                    }
                }
                dataMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold.setRows(dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter);
                resultMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold.setRows(dataMatrixWithoutSeparationAttributeAndValuesLowerThanThresholdRowCounter);
                dataMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold.setRows(dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter);
                resultMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold.setRows(dataMatrixWithoutSeparationAttributeAndValuesHigherThanThresholdRowCounter);
                newNode.setLeftChild(generateTree(dataMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold,resultMatrixWithoutSeparationAttributeAndValuesLowerThanThreshold));
                newNode.setRightChild(generateTree(dataMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold,resultMatrixWithoutSeparationAttributeAndValuesHigherThanThreshold));
                return newNode;
            }
        }

    }

    public Tree<Pair<NodeType,Double>> getDecisionTree(){
        Tree<Pair<NodeType,Double>> decisionTree = new Tree<>();
        decisionTree.setRoot(generateTree(repo.getDataMatrix(),repo.getResultsMatrix()));
        return decisionTree;
    }

    public Double evaluateDecisionTreeAccuracy(Tree<Pair<NodeType,Double>> decisionTree,Matrix<Double> data,Matrix<Double> results){
        Double numberOfCorrectClassifications = 0.0;
        for(int i = 0; i< data.getRows(); i++) {
            Node<Pair<NodeType, Double>> current = decisionTree.getRoot();
            while(current.getData().getFirst().equals(NodeType.SeparationAttribute)){
                if(data.get(i,(int) Math.round(current.getData().getSecond()))<current.getThreshold()){
                    current = current.getLeftChild();
                }
                else current = current.getRightChild();
            }
            if(current.getData().getSecond().equals(results.get(i,0))){
                numberOfCorrectClassifications++;
            }
        }
        return numberOfCorrectClassifications/(double) data.getRows();
    }

    public Double evaluateDecisionTreePerformance(Tree<Pair<NodeType,Double>> decisionTree,Matrix<Double> data,Matrix<Double> results){
        Double numberOfCorrectSpondylolisthesisClassifications = 0.0;
        for(int i = 0; i< data.getRows(); i++) {
            Node<Pair<NodeType, Double>> current = decisionTree.getRoot();
            while(current.getData().getFirst().equals(NodeType.SeparationAttribute)){
                if(data.get(i,(int) Math.round(current.getData().getSecond()))<current.getThreshold()){
                    current = current.getLeftChild();
                }
                else current = current.getRightChild();
            }
            if(current.getData().getSecond().equals(results.get(i,0)) && results.get(i,0)==0.5){
                numberOfCorrectSpondylolisthesisClassifications++;
            }
        }
        return numberOfCorrectSpondylolisthesisClassifications/(double) data.getRows();
    }

}
