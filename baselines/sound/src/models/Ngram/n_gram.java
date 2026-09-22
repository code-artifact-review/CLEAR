import java.io.File;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;

import slp.core.counting.giga.GigaCounter;
import slp.core.lexing.Lexer;
import slp.core.lexing.runners.LexerRunner;
import slp.core.lexing.simple.WhitespaceLexer;
import slp.core.modeling.Model;
import slp.core.modeling.dynamic.CacheModel;
import slp.core.modeling.mix.MixModel;
import slp.core.modeling.ngram.JMModel;
import slp.core.modeling.runners.ModelRunner;
import slp.core.translating.Vocabulary;

public class n_gram
{
    public static String dataset_root_dir = "../../../../../dataset/";
    public static String result_root_dir = "./n_gram_result/";

    public static String glance_all_dataset[] = {
        "ambari", "amq", "bookkeeper", "calcite", "cassandra", "flink",
        "groovy", "hbase", "hive", "ignite", "log4j2", "mahout", "mng",
        "nifi", "nutch", "storm", "tika", "ww", "zookeeper"
    };

    public static String glance_all_train_releases[] = {
        "ambari-1.2.0", "amq-5.0.0", "bookkeeper-4.0.0",
        "calcite-1.6.0", "cassandra-0.7.4", "flink-1.4.0",
        "groovy-1.0", "hbase-0.94.1", "hive-0.14.0",
        "ignite-1.0.0", "log4j2-2.0", "mahout-0.3",
        "mng-3.1.0", "nifi-0.4.0", "nutch-1.1", "storm-0.9.0",
        "tika-0.7", "ww-2.0.0", "zookeeper-3.4.6"
    };

    public static String glance_all_eval_releases[][] = {
        {"ambari-2.1.0", "ambari-2.2.0", "ambari-2.4.0", "ambari-2.5.0", "ambari-2.6.0", "ambari-2.7.0"},
        {"amq-5.1.0", "amq-5.2.0", "amq-5.4.0", "amq-5.5.0", "amq-5.6.0", "amq-5.7.0", "amq-5.8.0", "amq-5.9.0", "amq-5.10.0", "amq-5.11.0", "amq-5.12.0", "amq-5.14.0", "amq-5.15.0"},
        {"bookkeeper-4.2.0", "bookkeeper-4.4.0"},
        {"calcite-1.8.0", "calcite-1.11.0", "calcite-1.13.0", "calcite-1.15.0", "calcite-1.16.0", "calcite-1.17.0", "calcite-1.18.0"},
        {"cassandra-0.8.6", "cassandra-1.0.9", "cassandra-1.1.6", "cassandra-1.1.11", "cassandra-1.2.11"},
        {"flink-1.6.0"},
        {"groovy-1.5.5", "groovy-1.6.0", "groovy-1.7.3", "groovy-1.7.6", "groovy-1.8.1", "groovy-1.8.7", "groovy-2.1.0", "groovy-2.1.6", "groovy-2.4.4", "groovy-2.4.6", "groovy-2.4.8", "groovy-2.5.0", "groovy-2.5.5"},
        {"hbase-0.94.5", "hbase-0.98.0", "hbase-0.98.5", "hbase-0.98.11"},
        {"hive-1.2.0", "hive-2.0.0", "hive-2.1.0"},
        {"ignite-1.4.0", "ignite-1.6.0"},
        {"log4j2-2.1", "log4j2-2.2", "log4j2-2.3", "log4j2-2.4", "log4j2-2.5", "log4j2-2.6", "log4j2-2.7", "log4j2-2.8", "log4j2-2.9", "log4j2-2.10"},
        {"mahout-0.4", "mahout-0.5", "mahout-0.6", "mahout-0.7", "mahout-0.8"},
        {"mng-3.2.0", "mng-3.3.0", "mng-3.5.0", "mng-3.6.0"},
        {"nifi-1.2.0", "nifi-1.5.0", "nifi-1.8.0"},
        {"nutch-1.3", "nutch-1.4", "nutch-1.5", "nutch-1.6", "nutch-1.7", "nutch-1.8", "nutch-1.9", "nutch-1.10", "nutch-1.12", "nutch-1.13", "nutch-1.14", "nutch-1.15"},
        {"storm-0.9.3", "storm-1.0.0", "storm-1.0.3", "storm-1.0.5"},
        {"tika-0.8", "tika-0.9", "tika-0.10", "tika-1.1", "tika-1.3", "tika-1.5", "tika-1.7", "tika-1.10", "tika-1.13", "tika-1.15", "tika-1.17"},
        {"ww-2.0.5", "ww-2.0.10", "ww-2.1.1", "ww-2.1.3", "ww-2.1.7", "ww-2.2.0", "ww-2.2.2", "ww-2.3.1", "ww-2.3.4", "ww-2.3.10", "ww-2.3.15", "ww-2.3.17", "ww-2.3.20", "ww-2.3.24"},
        {"zookeeper-3.5.1", "zookeeper-3.5.2", "zookeeper-3.5.3"}
    };

    public static String linedp_all_dataset[] = {
        "activemq", "camel", "derby", "groovy", "hbase",
        "hive", "jruby", "lucene", "wicket"
    };

    public static String linedp_all_train_releases[] = {
        "activemq-5.0.0",
        "camel-1.4.0",
        "derby-10.2.1.6",
        "groovy-1_5_7",
        "hbase-0.94.0",
        "hive-0.9.0",
        "jruby-1.1",
        "lucene-2.3.0",
        "wicket-1.3.0-incubating-beta-1"
    };

    public static String linedp_all_eval_releases[][] = {
        {"activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"},
        {"camel-2.9.0", "camel-2.10.0", "camel-2.11.0"},
        {"derby-10.3.1.4", "derby-10.5.1.1"},
        {"groovy-1_6_BETA_1", "groovy-1_6_BETA_2"},
        {"hbase-0.95.0", "hbase-0.95.2"},
        {"hive-0.10.0", "hive-0.12.0"},
        {"jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"},
        {"lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"},
        {"wicket-1.3.0-beta2", "wicket-1.5.3"}
    };

    public static ModelRunner train_model(
        String dataset_name,
        String train_release
    )
    {
        String root_dir = (
            dataset_root_dir
            + dataset_name
            + "/n_gram_data/"
        );

        File train = new File(
            root_dir + train_release + "/src"
        );
        Lexer lexer = new WhitespaceLexer();
        LexerRunner lexerRunner = new LexerRunner(
            lexer,
            false
        );

        lexerRunner.setSentenceMarkers(true);

        Vocabulary vocabulary = new Vocabulary();

        Model model = new JMModel(
            6,
            new GigaCounter()
        );
        model = MixModel.standard(
            model,
            new CacheModel()
        );
        ModelRunner modelRunner = new ModelRunner(
            model,
            lexerRunner,
            vocabulary
        );
        modelRunner.learnDirectory(train);

        return modelRunner;
    }

    public static void predict_defective_lines(
        String dataset_name,
        String train_release,
        String test_release,
        ModelRunner modelRunner
    ) throws Exception
    {
        String root_dir = (
            dataset_root_dir
            + dataset_name
            + "/n_gram_data/"
        );
        String result_dir = (
            result_root_dir
            + dataset_name
            + "/"
        );

        File result_directory = new File(result_dir);
        if (!result_directory.exists()) {
            result_directory.mkdirs();
        }

        LexerRunner lexerRunner = (
            modelRunner.getLexerRunner()
        );

        StringBuilder sb = new StringBuilder();

        sb.append(
            "train-release,test-release,file-name,"
            + "line-number,token,token-score,line-score\n"
        );

        File test_java_dir = new File(
            root_dir + test_release + "/src/"
        );
        File java_files[] = test_java_dir.listFiles();

        String line_num_path = (
            root_dir
            + test_release
            + "/line_num/"
        );

        for (int j = 0; j < java_files.length; j++)
        {
            File test = java_files[j];

            String filename = test.getName();
            String filename_original = filename
                .replace("_", "/")
                .replace(".txt", ".java");
            String linenum_filename = filename
                .replace(".txt", "_line_num.txt");

            List<String> linenum = FileUtils.readLines(
                new File(
                    line_num_path + linenum_filename
                ),
                "UTF-8"
            );

            List<List<Double>> fileEntropies = (
                modelRunner.modelFile(test)
            );
            List<List<String>> fileTokens = lexerRunner
                .lexFile(test)
                .map(
                    line -> line.collect(
                        Collectors.toList()
                    )
                )
                .collect(Collectors.toList());

            for (int i = 0; i < linenum.size(); i++)
            {
                List<String> lineTokens = fileTokens.get(i);
                List<Double> lineEntropies = (
                    fileEntropies.get(i)
                );

                String cur_line_num = linenum.get(i);

                DoubleSummaryStatistics lineStatistics = (
                    lineEntropies.stream()
                        .mapToDouble(Double::doubleValue)
                        .summaryStatistics()
                );
                double averageEntropy = (
                    lineStatistics.getAverage()
                );

                for (int k = 0; k < lineTokens.size(); k++)
                {
                    String tok = lineTokens.get(k);
                    double tok_score = lineEntropies.get(k);

                    if (tok == "<s>")
                        continue;

                    sb.append(
                        train_release
                        + ","
                        + test_release
                        + ","
                        + filename_original
                        + ","
                        + cur_line_num
                        + ","
                        + tok
                        + ","
                        + tok_score
                        + ","
                        + averageEntropy
                        + "\n"
                    );
                }
            }
        }

        FileUtils.write(
            new File(
                result_dir
                + test_release
                + "-line-lvl-result.txt"
            ),
            sb.toString(),
            "UTF-8"
        );
    }

    public static void train_eval_model(
        String dataset_name,
        int dataset_idx,
        String all_dataset[],
        String all_train_releases[],
        String all_eval_releases[][]
    ) throws Exception
    {
        String project_name = all_dataset[dataset_idx];

        String initial_train_release = (
            all_train_releases[dataset_idx]
        );

        String eval_releases[] = (
            all_eval_releases[dataset_idx]
        );


        String release_sequence[] = new String[
            eval_releases.length + 1
        ];

        release_sequence[0] = initial_train_release;

        for (
            int idx = 0;
            idx < eval_releases.length;
            idx++
        )
        {
            release_sequence[idx + 1] = (
                eval_releases[idx]
            );
        }

        for (
            int idx = 0;
            idx < release_sequence.length - 1;
            idx++
        )
        {
            String train_release = (
                release_sequence[idx]
            );
            String test_release = (
                release_sequence[idx + 1]
            );

            ModelRunner modelRunner = train_model(
                dataset_name,
                train_release
            );

            System.out.println(
                "Finish training model for "
                + dataset_name
                + " / "
                + project_name
                + " / "
                + train_release
            );

            predict_defective_lines(
                dataset_name,
                train_release,
                test_release,
                modelRunner
            );

            System.out.println(
                "Finish "
                + dataset_name
                + " / "
                + train_release
                + " -> "
                + test_release
                + "\n"
            );
        }
    }

    public static void train_eval_dataset(
        String dataset_name,
        String all_dataset[],
        String all_train_releases[],
        String all_eval_releases[][]
    ) throws Exception
    {
        for (
            int dataset_idx = 0;
            dataset_idx < all_dataset.length;
            dataset_idx++
        )
        {
            train_eval_model(
                dataset_name,
                dataset_idx,
                all_dataset,
                all_train_releases,
                all_eval_releases
            );
        }
    }

    public static void main(
        String[] args
    ) throws Exception
    {
        //train_eval_dataset(
        //    "glance_dataset",
        //    glance_all_dataset,
        //    glance_all_train_releases,
        //    glance_all_eval_releases
        //);

        train_eval_dataset(
            "linedp_dataset",
            linedp_all_dataset,
            linedp_all_train_releases,
            linedp_all_eval_releases
        );
    }
}
