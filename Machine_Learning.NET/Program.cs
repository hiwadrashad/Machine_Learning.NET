using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Shared.DTO;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Machine_Learning.NET
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======================================== MACHINE LEARNING ================================");
            Console.WriteLine("\n");
            Console.WriteLine("Generate machine learning object");
            var mlContext = new MLContext();
            var trainingData = new List<MoviePreferenceInput>();
            Console.WriteLine("\n");
            Console.WriteLine("Generating preference models");
            var dieHardLover = new MoviePreferenceInput
            {
                StarWarsScore = 8,
                ArmageddonScore = 10,
                SleeplessInSeattleScore = 1,
                ILikeDieHard = true
            };
            var dieHardHater = new MoviePreferenceInput
            {
                StarWarsScore = 1,
                ArmageddonScore = 1,
                SleeplessInSeattleScore = 9,
                ILikeDieHard = false
            };
            Console.WriteLine("\n");
            Console.WriteLine("Populate data");
            for (var i = 0; i < 100; i++)
            {
                trainingData.Add(dieHardLover);
                trainingData.Add(dieHardHater);
            }

            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);
            ITransformer model;
            Console.WriteLine("\n");
            Console.WriteLine("Train or retrain ML.NET dependant on if previous data exists");
            if (!File.Exists("./diehard-model.zip") && !File.Exists("./diehard-pipeline.zip"))
            {
                model = Shared.MachineLearning.MLMethods.TrainNewModel(mlContext, trainingDataView);
            }
            else
            {
                model = Shared.MachineLearning.MLMethods.RetrainModel(mlContext, trainingDataView);
            }
            Console.WriteLine("\n");
            Console.WriteLine("Generate models to predict their preference");
            var input1 = new MoviePreferenceInput
            {
                StarWarsScore = 7,
                ArmageddonScore = 9,
                SleeplessInSeattleScore = 0
            };
            var input2 = new MoviePreferenceInput
            {
                StarWarsScore = 0,
                ArmageddonScore = 0,
                SleeplessInSeattleScore = 10
            };
            Console.WriteLine("\n");
            Console.WriteLine("Calculate predictions");
            Console.WriteLine("\n");
            PredictionEngine<MoviePreferenceInput, PreferencePrediction> predictionEngine =
                mlContext.Model.CreatePredictionEngine<MoviePreferenceInput, PreferencePrediction>(model);
            var prediction = predictionEngine.Predict(input1);
            Console.WriteLine($"First user loves Die Hard: {prediction.Prediction}");
            prediction = predictionEngine.Predict(input2);
            Console.WriteLine("\n");
            Console.WriteLine($"Second user loves Die Hard: {prediction.Prediction}");
        }

    }

}
