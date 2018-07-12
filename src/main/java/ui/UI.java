package ui;

import model.NodeType;
import model.Pair;
import model.Tree;
import repository.FileRepo;
import service.DecisionTreesService;


import java.util.Scanner;

public class UI {

    private DecisionTreesService decisionTreesService;

    private Scanner sc;
    public UI(){
        sc = new Scanner(System.in);
    }

    /**
     * Displays the main menu of the application
     */
    public void displayMainMenu() {

        FileRepo repo = new FileRepo("column_3C_weka_data.arff", "column_3C_weka_test.arff");
        decisionTreesService = new DecisionTreesService(3,repo);
        Tree<Pair<NodeType,Double>> decisionTree=null;

        Boolean merge = true;
        while(merge) {

            int x;
            System.out.println("1.Formare arbore de decizie.");
            System.out.println("2.Testare date de antrenare.");
            System.out.println("3.Testare date de testare.");
            System.out.println("4.Iesire.");
            x = sc.nextInt();


            switch (x) {




                case 1: {
                    System.out.println();
                    decisionTree = decisionTreesService.getDecisionTree();
                    System.out.println("Arborele de decizie a fost generat cu succes!");
                    System.out.println();
                    break;
                }
                case 2: {
                    System.out.println();
                    System.out.println("Acuratetea este: "+decisionTreesService.evaluateDecisionTreeAccuracy(decisionTree,repo.getDataMatrix(),repo.getResultsMatrix())*100+"%");
                    System.out.println("Performanta pentru detectarea spoliozei este: "+decisionTreesService.evaluateDecisionTreePerformance(decisionTree,repo.getDataMatrix(),repo.getResultsMatrix())*100+"%");
                    System.out.println();
                    break;
                }

                case 3: {
                    System.out.println();
                    System.out.println("Acuratetea este: "+decisionTreesService.evaluateDecisionTreeAccuracy(decisionTree,repo.getTestDataMatrix(),repo.getTestResultsMatrix())*100+"%");
                    System.out.println("Performanta pentru detectarea spoliozei este: "+decisionTreesService.evaluateDecisionTreePerformance(decisionTree,repo.getTestDataMatrix(),repo.getTestResultsMatrix())*100+"%");
                    System.out.println();
                    break;
                }
                case 4: {
                    merge = false;
                    break;
                }

                default:
                    System.out.println("Optiune invalida! Mai incercati!");
            }
        }


    }
}
