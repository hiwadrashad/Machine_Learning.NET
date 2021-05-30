using System;
using System.Collections.Generic;
using System.Text;

namespace Shared.DTO
{
    public class MoviePreferenceInput
    {
        public float StarWarsScore { get; set; }
        public float ArmageddonScore { get; set; }
        public float SleeplessInSeattleScore { get; set; }
        public bool ILikeDieHard { get; set; }
    }
}
