package hex.infogram;

import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.fvec.Vec;
import water.parser.ParseSetup;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.DMatrix.transpose;
import static hex.infogram.InfoGramModel.InfoGramParameter;
import static hex.infogram.InfoGramModel.InfoGramParameter.Algorithm;

public class InfoGramPipingTest extends TestUtil {
  public static final double TOLERANCE = 1e-6;
  @BeforeClass public static void setup() { stall_till_cloudsize(1); }
  
  // Deep example 9
  @Test
  public void testIris() {
    try {
      Scope.enter();
      List<String> predictorNames = new ArrayList<>(Arrays.asList("sepal_len", "sepal_wid", "petal_len", "petal_wid"));
      double[] deepRel = new double[]{0.009862314, 0.012677615, 1.000000000, 0.993662592};
      double[] deepCMI = new double[]{0.1022320, 0.7271748, 0.5640882, 1.0000000};
      Frame trainF = parseTestFile("smalldata/iris/iris_wheader.csv");
      Scope.track(trainF);
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "class";
      params._train = trainF._key;
      params._infogram_algorithm = Algorithm.gbm;
      params._ntop = 5;
      params._seed = 12345;

      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      assertEqualCMIRel(predictorNames, deepRel, deepCMI, infogramModel._output, TOLERANCE);
      Frame infogramFrame = DKV.getGet(infogramModel._output._cmiRelKey);
      Scope.track(infogramFrame);
      // Erin:  please uncomment with correct path that you want to save the csv file.
/*      TestUtil.writeFrameToCSV("/pathToyourFile/infogramIris.csv", infogramFrame, true,false);
    } catch (IOException e) {
      e.printStackTrace();*/
    } finally {
      Scope.exit();
    }
  }
  
  // Deep example 5
  @Test
  public void testGermanData() {
    try {
      Scope.enter();
      double[] deepRel = new double[]{1.00000000, 0.58302027, 0.43431236, 0.66177924, 0.53677082, 0.25084764, 
              0.34379833, 0.13251726, 0.11473028, 0.09548423, 0.20398740, 0.16432640, 0.06875276, 0.04870468, 
              0.12573930, 0.01382682, 0.04496173, 0.01273963};
      double[] deepCMI = new double[]{0.84946975, 0.73020930, 0.58553936, 0.75780528, 1.00000000, 0.38461582, 
              0.57575695, 0.30663930, 0.07604779, 0.19979514, 0.42293369, 0.20628365, 0.25316918, 0.15096705, 
              0.24501686, 0.11296778, 0.13068605, 0.03841617};
      Frame trainF = parseTestFile("smalldata/admissibleml_test/german_credit.csv");
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "BAD";
      trainF.replace(trainF.numCols()-1, trainF.vec(params._response_column).toCategoricalVec()).remove();
      DKV.put(trainF);
      params._train = trainF._key;
      params._infogram_algorithm = Algorithm.gbm;
      params._sensitive_attributes = new String[]{"status_gender", "age"};
      params._ntop = 50;
      Scope.track(trainF);

      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      List<String> predictorNames = new ArrayList<>(Arrays.asList(infogramModel._output._all_predictor_names));
      assertEqualCMIRel(predictorNames, deepRel, deepCMI, infogramModel._output, TOLERANCE);
      Frame infogramFr = DKV.getGet(infogramModel._output._cmiRelKey);
      Scope.track(infogramFr);
      // **** ERIN:  please put in correct path and uncomment the following.
/*      TestUtil.writeFrameToCSV("/ErinPleaseAddPath/infogramGermanData.csv", infogramFr, true, false);
    } catch (IOException e) {
      e.printStackTrace();*/
    } finally {
      Scope.exit();
    }
  }

  // Deep example 6
  @Test
  public void testUCICredit() {
    try {
      Scope.enter();
      List<String> predictorNames = new ArrayList<>(Arrays.asList("X1","X3","X4","X6","X7","X8","X9","X10","X11","X12", 
              "X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23"));
      double[] deepCMI = new double[]{0.39946282, 0.05979470, 0.03750023, 0.92207648, 0.60110485, 0.47026827,
              0.42284063, 0.38576886, 0.34166255, 1.00000000, 0.71440922, 0.84539821, 0.73648997, 0.80649469,
              0.85364615, 0.72847950, 0.68435443, 0.69698726, 0.69815672, 0.61204628, 0.63450797};
      double[] deepRel = new double[]{0.224428830, 0.012273940, 0.008153018, 1.000000000, 0.176288025,
              0.056892118, 0.032841546, 0.032949924, 0.033377301, 0.123672983, 0.048615394, 0.020703590, 0.023215394,
              0.023312163, 0.024406971, 0.031799996, 0.040570292, 0.058936513, 0.046192408, 0.008947963, 0.031327136};
      //Frame trainF = parseTestFile("smalldata/admissibleml_test/UCICreditCardData.csv");
      Frame trainF = parseTestFile("smalldata/admissibleml_test/UCICreditCardData.csv", null,
              ParseSetup.HAS_HEADER, new byte[]{4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4});
      int ncol = trainF.numCols();
      for (int cInd = 0; cInd < ncol; cInd++) {
        trainF.replace(cInd, trainF.vec(cInd).toCategoricalVec()).remove();
      }
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "Y";
      Scope.track(trainF.remove(0));
      DKV.put(trainF);
      Scope.track(trainF);
      params._train = trainF._key;
      params._infogram_algorithm = Algorithm.gbm;
      params._model_algorithm = Algorithm.gbm;
      params._seed = 12345;
      params._sensitive_attributes = new String[]{"X2", "X5"};

      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      assertEqualCMIRel(predictorNames, deepRel, deepCMI, infogramModel._output, TOLERANCE);
      Frame infogramFr = DKV.getGet(infogramModel._output._cmiRelKey);
      Scope.track(infogramFr);
      // **** ERIN:  please put in correct path and uncomment the following.
/*      TestUtil.writeFrameToCSV("/ErinPleaseAddPath/infogramUCICredit.csv", infogramFr, true, false);
    } catch (IOException e) {
      e.printStackTrace();*/
    } finally {
      Scope.exit();
    }
  }

  // Deep example 10
  @Test
  public void testProstateWide() {
    try {
      Scope.enter();
      double[] deepCMI = new double[]{1.000000000, 0.416069511, 0.685096529, 0.507368744, 0.000000000, 0.575576639, 
              0.000000000, 0.000000000, 0.271674757, 0.248421424, 0.474729852, 0.000000000, 0.000000000, 0.000000000, 
              0.018534010, 0.036368120, 0.000000000, 0.184383526, 0.316296264, 0.239501471, 0.000000000, 0.000000000, 
              0.058800272, 0.000000000, 0.073573180, 0.396318927, 0.267592247, 0.000000000, 0.000000000, 0.000000000,
              0.000000000, 0.041711020, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.002055904, 
              0.000000000, 0.000000000, 0.000000000, 0.090010963, 0.048595528, 0.080272375, 0.004002294, 0.000000000, 
              0.061638596, 0.000000000, 0.087287940, 0.062288766};
      double[] deepRel = new double[]{1.000000e+00, 6.486737e-01, 4.947306e-01, 3.642898e-01, 1.932296e-01,
              1.748593e-01, 1.067082e-01, 9.285556e-02, 9.007357e-02, 6.817226e-02, 5.514745e-02, 5.039264e-02,
              4.741128e-02, 4.188043e-02, 4.024860e-02, 3.019085e-02, 2.663353e-02, 2.570307e-02, 2.379470e-02, 
              2.220692e-02, 2.044017e-02, 1.524100e-02, 1.663565e-02, 1.111248e-02, 7.832062e-03, 6.588439e-03, 
              5.934081e-03, 3.756785e-03, 1.789985e-03, 2.941161e-03, 4.941466e-05, 2.269027e-03, 2.224223e-03, 
              1.902141e-03, 2.644488e-06, 8.339014e-04, 1.352506e-03, 1.371541e-03, 1.290422e-03, 7.044505e-04, 
              1.499066e-03, 2.239000e-04, 6.416308e-04, 7.102487e-04, 1.212104e-03, 5.354030e-04, 5.900479e-04, 
              6.537474e-04, 5.808857e-04, 3.335511e-04};
      Frame trainF = Scope.track(parseTestFile("smalldata/admissibleml_test/prostmat.csv"));
      Frame transposeF = new Frame(transpose(trainF));
      double[] y = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1};
      Vec responseVec = Vec.makeVec(y, Vec.newKey());
      Scope.track(responseVec);
      transposeF.add("y",responseVec);
      transposeF.replace(transposeF.numCols()-1, transposeF.vec("y").toCategoricalVec()).remove();
      Scope.track(transposeF);
      DKV.put(transposeF);
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "y";
      params._train = transposeF._key;
      params._infogram_algorithm = Algorithm.gbm;
      params._ntop = 50;
      params._seed = 12345;

      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      List<String> predictorNames = new ArrayList<>(Arrays.asList(infogramModel._output._all_predictor_names));
      assertEqualCMIRel(predictorNames, deepRel, deepCMI, infogramModel._output, TOLERANCE);
      // **** ERIN:  please put in correct path and uncomment the following.
/*      TestUtil.writeFrameToCSV("/ErinPleaseAddPath/infogramProstate.csv", infogramFr, true, false);
    } catch (IOException e) {
      e.printStackTrace();*/
    } finally {
      Scope.exit();
    }
  }

  // Deep example 12
  @Test
  public void testCompasScores2years() {
    try {
      Scope.enter();
      double[] deepCMI = new double[]{0.008011609, 0.138541125, 0.013504090,0.016311791, 0.181746765, 0.046041515, 
              0.017965106, 0.138541125, 0.107449216, 0.082824953, 0.050455340, 0.181746765, 0.050691145, 1.000000000,
              0.858833177};
      double[] deepRel = new double[]{0.0013904297, 0.0167956478, 0.0022520089, 0.0014052132, 0.0104887361, 
              0.0041130581, 0.0012933420, 0.0000000000, 0.0018857971, 0.0040163313, 0.0002574921, 0.0000000000, 
              0.0145265544, 1.0000000000, 0.2223278895};
      Frame trainF = Scope.track(parseTestFile("smalldata/admissibleml_test/compas_full.csv"));
      trainF.replace(trainF.numCols()-1, trainF.vec("two_year_recid").toCategoricalVec()).remove();
      DKV.put(trainF);
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "two_year_recid";
      params._train = trainF._key;
      params._sensitive_attributes = new String[]{"sex","age","race"};
      params._infogram_algorithm = Algorithm.gbm;
      params._ignored_columns = new String[]{"id"};
      params._ntop = 50;
      params._seed = 12345;

      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
      List<String> predictorNames = new ArrayList<>(Arrays.asList(infogramModel._output._all_predictor_names));
      assertEqualCMIRel(predictorNames, deepRel, deepCMI, infogramModel._output, TOLERANCE);
      // **** ERIN:  please put in correct path and uncomment the following.
/*      TestUtil.writeFrameToCSV("/ErinPleaseAddPath/infogramCompas.csv", infogramFr, true, false);
    } catch (IOException e) {
      e.printStackTrace();*/
    } finally {
      Scope.exit();
    }
  }
  
  @Test
  public void testInfoGramInvoke() {
    try {
      Scope.enter();
      Frame trainF = parseTestFile("smalldata/glm_test/binomial_20_cols_10KRows.csv");
      convertCols(trainF);  // convert integer columns to enum columns
      Scope.track(trainF);
      DKV.put(trainF);
      
      InfoGramParameter params = new InfoGramParameter();
      params._response_column = "C21";
      params._train = trainF._key;
      params._infogram_algorithm = Algorithm.gbm;
      params._infogram_algorithm_params = "{\"sample_rate\" : [0.3], \"col_sample_rate\" : [0.3]}";
      params._parallelism = 4;
      params._ntop = 19;
      
      InfoGramModel infogramModel = new InfoGram(params).trainModel().get();
      Scope.track_generic(infogramModel);
    } finally {
      Scope.exit();
    }
  }
  
  public static void convertCols(Frame train) {
    final int numCols = train.numCols();
    final int enumCols = (numCols-1)/2;
    for (int index=0; index < enumCols; index++)
      train.replace(index, train.vec(index).toCategoricalVec()).remove();
    final int responseIndex = numCols-1;
    train.replace(responseIndex, train.vec(responseIndex).toCategoricalVec()).remove();
  }

  public static void assertEqualCMIRel(List<String> predictorNames, double[] deepRel, double[] deepCMI,
                                       InfoGramModel.InfoGramOutput output, double tolerance) {
    int numPred = predictorNames.size();
    String[] predictorWNames = output._all_predictor_names;
    double[] modelRelevance = output._relevance;
    double[] modelCMI = output._cmi_normalize;
    for (int index = 0; index < numPred; index++) {
      // compare relevance with deep result
      String predName = predictorWNames[index];
      int predIndex = predictorNames.indexOf(predName);
      assert Math.abs(modelRelevance[index]-deepRel[predIndex])<tolerance : "model relevance "+
              modelRelevance[index]+" and deep relevance "+deepRel[predIndex]+" for predictor "
              +predName + " differs more than "+tolerance;
      // compare CMI with deep result
      assert Math.abs(modelCMI[index]-deepCMI[predIndex])<tolerance : "model CMI "+
              modelCMI[index]+" and deep CMI "+deepCMI[predIndex]+" for predictor "
              +predName + " differs more than "+tolerance;
    }
  }
}
