using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Shared.DTO
{
    public class PreferencePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
