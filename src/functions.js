/* FreshPastaCalculator - Custom Functions */

/* ------------------------- */
/* Utility Functions         */
/* ------------------------- */

function take(arr, n){
  return arr.slice(0, n);
}

function sum(arr){
  return arr.reduce((a, b) => a + b, 0);
}

function sumproduct(a, b){
  return a.reduce((acc, v, i) => acc + v * b[i], 0);
}

function clampScalar(x, lo, hi){
  return Math.min(Math.max(x, lo), hi);
}

function clamp(x, lo, hi){
  if (Array.isArray(x))
    return x.map(v => clampScalar(v, lo, hi));

  return clampScalar(x, lo, hi);
}

function sigmoid(lo, hi, mid, alpha, x){
  return lo + (hi - lo) / (1 + Math.exp(-alpha * (x - mid)));
}

function linear(lo, hi, x){
  return lo + (hi - lo) * x;
}

function arrhenius(A, T, Tref = null){
  const k = T => Math.exp(-A / (8.314 * (T + 273.15)));
  if (Tref === null)
    return k(T);

  return k(T) / k(Tref);
}

function moistureDryToWet(x){
  return 1 / (1 / x + 1);
}

/* ------------------------- */
/* Lookup Helpers            */
/* ------------------------- */

function choose(arr, idx){
  return arr[idx - 1];
}

function xmatch(value, array){
  return array.indexOf(value) + 1;
}

function lookup(value, lookupArray, returnArray, defaultValue = 0){
  const idx = lookupArray.indexOf(value);
  return (idx >= 0? returnArray[idx]: defaultValue);
}

/* ------------------------- */
/* Utility helpers           */
/*-------------------------- */

// Normalize flour fractions (private helper)
// Input: fractionsRaw = array of numbers
// Output: normalized array summing to 1
function normalizeFractions(fractionsRaw) {
  // Count only numeric, non-null, non-NaN entries
  const flours = fractionsRaw.filter(v => typeof v === "number" && !isNaN(v)).length;
  // Take only the first "flours" entries
  const fractions = fractionsRaw.slice(0, flours);
  const total = fractions.reduce((a, b) => a + b, 0);
  return fractions.map(v => v / total);
}

function bisection(f, comp, a, b, iterations){
  let lo = a;
  let hi = b;
  for (let i = 0; i < iterations; i ++){
    const mid = (lo + hi) / 2;
    if (f(mid) > comp)
      hi = mid;
    else
      lo = mid;
  }
  return (lo + hi) / 2;
}

function softplus(x, k){
  return Math.log(1 + Math.exp(k * x)) / k;
}

function expDecay(k, x){
  return Math.exp(-k * x);
}

function expRise = (lo, hi, k, x){
  return lo + (hi - lo) * (1 - Math.exp(-k * x));
}

function diffusionTime = (L, D, reduction){
  const tc = Math.pow(L / Math.PI, 2) / D;
  return tc * Math.log(8 / (Math.pow(Math.PI, 2) * (1 - reduction)));
}

function diffusionFraction = (L, D, t){
  return 1 - (8 / Math.pow(Math.PI, 2)) * Math.exp(-D * t * Math.pow(Math.PI / L, 2));
}

function lookup = (value, lookupArray, returnArray, defaultValue = 0){
  const idx = lookupArray.indexOf(value);
  return (idx >= 0? returnArray[idx]: defaultValue);
}


/* ------------------------- */
/* CONSTANTS                 */
/* ------------------------- */

const flourTypes = [
  "wheat","wheat semolina","wheat semolina fine",
  "durum wheat","durum wheat semolina","durum wheat semolina fine",
  "rye",
  "einkorn","emmer","spelt",
  "buckwheat",
  "barley",
  "chestnut"
];
const pastaTypes = ["barley malt (diastatic)", "barley malt (non-diastatic)"];
const processes  = ["laminated", "screw extruded", "piston extruded"];
const usageTypes = ["filled", "unfilled"];
const dieTypes   = ["brass", "teflon"];

const albumenProtein = 0.103;
const albumenWater = 0.881;
const albumenFat = 0.002;
const yolkProtein = 0.15;
const yolkWater = 0.587;
const yolkFat = 0.26;


/* ------------------------- */
/* MAIN FUNCTIONS            */
/* ------------------------- */

/**
 * FPC_MOISTURE_MODEL
 * Computes flour moisture and strictly bound water for a flour blend.
 *
 * @customfunction
 */
export function FPC_MOISTURE_MODEL(
  fractionsRaw,
  flourMatrix,
  flourTemperature,
  airRelativeHumidity
){
  const normalized = normalizeFractions(fractionsRaw);

  const col = index => take(flourMatrix.map(r => r[index]), flours);
  const flourStrength = col(0);
  const flourSugar    = col(3);
  const flourProtein  = col(4);
  const flourFiber    = col(6);
  const flourAsh      = col(7);
  const flourType     = col(9);

  const dot = v => sumproduct(v, normalized);

  const fidx = flourType.map(t => flourTypes.indexOf(t) + 1);

  const strictlyBoundWaterDB = flourProtein.map((fp, i) => 0.045 + 0.0025 * fp + 0.015 * flourFiber[i]);

  // GAB parameters
  const αC = flourProtein.map((fp, i) => 10775.5 + 428.6 * fp + 918.9 * flourFiber[i]);
  const αK = flourProtein.map((fp, i) => 2417.5 + 171.8 * fp + 307.6 * flourFiber[i]);
  const C = flourProtein.map((fp, i) => (11.45 + 0.366 * fp - 1.285 * flourFiber[i] + 0.848 * flourAsh[i]) * arrhenius(αC[i], flourTemperature, 20));
  const K = flourProtein.map((fp, i) => (1.09 - 0.0083 * fp - 0.0536 * flourFiber[i] - 0.0156 * flourAsh[i]) * arrhenius(αK[i], flourTemperature, 20));
  const n = flourProtein.map((fp, i) => 1.4 - 0.0204 * fp + 0.04 * flourFiber[i] + 0.00026 * flourAsh[i]);
  const waterActivityTerm = K.map(k => clamp(k * airRelativeHumidity, 0, 0.95));
  const boundWaterDB = strictlyBoundWaterDB.map((sbw, i) =>{
    const C_i = C[i];
    const n_i = n[i];
    const t = waterActivityTerm[i];
    const numerator = C_i * t * Math.pow(1 + (n_i - 1) * t, n_i - 1);
    const denominator = (1 - t) * Math.pow(1 - (1 - C_i) * t, n_i);
    return sbw * Math.max(1, numerator / denominator);
  });

  return [
    [dot(boundWaterDB)],
    [dot(strictlyBoundWaterDB)]
  ];
}

/**
 * FPC_EGG_SOLVER
 * Computes fresh pasta water, egg amount, additional water and final protein.
 * This is a full translation of your LET-based Excel model.
 *
 * @customfunction
 */
export function FPC_EGG_SOLVER(
  fractionsRaw,
  flourMatrix,
  flourMatrix,
  airRelativeHumidity,
  pastaType,
  pastaThickness,
  pastaProcess,
  dieMaterial,
  dieThickness,
  pastaUsageType,
  targetProtein,
  oil,
  salt,
  barleyMalt,
  flourMoistureDB,
  strictlyBoundWaterDB,
  albumen,
  yolk
){

  const normalized = normalizeFractions(fractionsRaw);

  const col = index => take(flourMatrix.map(r => r[index]), flours);
  const flourStrength = col(0);
  const flourSugar    = col(3);
  const flourProtein  = col(4);
  const flourFiber    = col(6);
  const flourAsh      = col(7);
  const flourType     = col(9);

  const dot = v => sumproduct(v, normalized);

  const fidx = flourType.map(t => xmatch(t, flourTypes));
  const bmidx = xmatch(pastaType, pastaTypes);
  const pidx  = xmatch(pastaProcess, processes);
  const puidx = xmatch(pastaUsageType, usageTypes);
  const dmidx = xmatch(dieMaterial, dieTypes);

  const baseDamaged = [0.029, 0.038, 0.033, 0.036, 0.032, 0.035, 0.023, 0.026, 0.028, 0.030, 0.019, 0.024, 0.017];
  const damagedStarch = fidx.map((idx, i) =>{
    const base = baseDamaged[idx - 1];
    const hard = sigmoid(0, 1, 0.5, 10,
      idx <= 2
        ? (flourStrength[i] - 180) / 220
        : (flourProtein[i] - 0.10) / 0.05
    );
    const extr = sigmoid(0, 1, 0.5, 10,
	   (flourAsh[i] - 0.005) / 0.007);
    const milling = 1 + 0.6 * (flourAsh[i] - 0.0055);

    const βH = 0.2 + 0.04 * (flourStrength[i] - 80) + 0.02 * (flourProtein[i] - 0.10);
    const βE = 0.2 + 0.15 * flourAsh[i] + 0.03 * flourFiber[i];
    const βI = 0.05 + 0.015 * flourProtein[i] + 0.02 * flourFiber[i];

    return clamp(
      (base + βH * hard + βE * extr + βI * hard * extr) * milling,
      0.015,
      0.06
    );
  });

  const flourGluten = [0.78, 0.83, 0.80, 0.83, 0.80, 0.75, 0.55, 0.73, 0.75, 0.77, 0.00, 0.08, 0.00];
  const glutenProtein = dot(flourProtein.map((fp, i) => fp * flourGluten[fidx[i] - 1]));
  const protein_k = 1 - 0.015 * Math.max(0, dot(flourProtein) - 0.11);
  const fiber_k = 1 - 0.02 * dot(flourFiber);
  const oilProcess_k = choose([1.1, 0.85, 0.95], pidx);
  const oilWater_k = sigmoid(0.18, 0.35, 0.25, 50,
    0.25 * protein_k * fiber_k * oilProcess_k * (glutenProtein / 0.12)
  );

  const flourMoistureEffective = moistureDryToWet(flourMoistureDB - strictlyBoundWaterDB);
  const nonEggWater = flourMoistureEffective + barleyMalt * (bmidx === 1? 0.012: 0.007) + oil * oilWater_k;

  const funFreshPastaWater = egg => choose([1, 1, 1], 1) * (1 + 0.04 * egg * (albumen / (albumen + yolk || 1))) * (1 - 0.03 * egg * (yolk / (albumen + yolk || 1)));
  const funAdditionalWater = egg => funFreshPastaWater(egg) - nonEggWater - egg * (albumenWater + yolkWater);
  const funPastaProtein = egg => (dot(flourProtein) + egg * (albumenProtein + yolkProtein)) / (1 + funAdditionalWater(egg) + egg);

  const egg_alpha = 0.04 * (albumen / (albumen + yolk || 1));
  const egg_beta  = 0.03 * (yolk / (albumen + yolk || 1));
  let eggMax = 0;
  if (albumenWater !== 0 || yolkWater !== 0){
    const coeff = 1;
    const nonEgg = nonEggWater;

    const a = -coeff * egg_alpha * egg_beta;
    const b = coeff * (egg_alpha - egg_beta) - (albumenWater + yolkWater);
    const c = coeff - nonEgg;

    if (egg_alpha === 0 && egg_beta === 0)
      eggMax = 0;
	 else if (egg_alpha === 0)
      eggMax = (nonEgg - coeff) / (coeff * egg_beta + (albumenWater + yolkWater));
	 else if (egg_beta === 0)
      eggMax = (nonEgg - coeff) / (coeff * egg_alpha - (albumenWater + yolkWater));
	 else
      eggMax = (-b - Math.sqrt(b * b - 4 * a * c)) / (2 * a);
  }
  eggMax = Math.max(0, eggMax);

  const egg = (albumen + yolk > 0)? bisection(funPastaProtein, targetProtein, 0, eggMax, 20): 0;

  return [
    [funFreshPastaWater(egg)],
    [egg],
    [funAdditionalWater(egg)],
    [funPastaProtein(egg)]
  ];
}

/**
 * FPC_OPTIMIZATION
 * Computes WMin, PLOpt, proteinMin, ashMax, oilOpt, saltOpt, barleyMaltAromaticOpt.
 * Direct translation of your LET-based Excel model.
 *
 * @customfunction
 */
export function FPC_OPTIMIZATION(
  fractionsRaw,
  flourMatrix,
  pastaType,
  pastaThickness,
  pastaProcess,
  dieMaterial,
  pastaUsageType,
  albumen,
  yolk,
  oil,
  salt,
  barleyMalt,
  additionalWater
){

  const normalized = normalizeFractions(fractionsRaw);

  const col = index => take(flourMatrix.map(r => r[index]), flours);
  const flourStrength = col(0);
  const flourProtein  = col(4);
  const flourFiber    = col(6);
  const flourAsh      = col(7);
  const flourType     = col(9);

  const dot = v => sumproduct(v, normalized);

  const fidx = flourType.map(t => xmatch(t, flourTypes));
  const bmidx = xmatch(pastaType, pastaTypes);
  const pidx  = xmatch(pastaProcess, processes);
  const puidx = xmatch(pastaUsageType, usageTypes);
  const dmidx = xmatch(dieMaterial, dieTypes);

  const glutenShare = dot(fidx.map(idx => choose([1,1,1,1,1,1,0,1,1,1,0,0,0], idx)));
  const pastaThickness_k = sigmoid(0, 1, 0.5, 10,
    (1 - pastaThickness) / 1.2);
  const dieMaterial_k = (dmidx === 1? 1: 0);
  const pastaUsageType_k = (puidx === 1? 1: 0);
  const WMin = sigmoid(0, 150 + 35 * dieMaterial_k + 30 * pastaUsageType_k + 40 * pastaThickness_k, 0.5, glutenShare, 20);

  const PLOpt = sigmoid(0.35, 0.45 + 0.1 * dieMaterial_k + 0.05 * pastaUsageType_k + 0.08 * pastaThickness_k, 0.5, glutenShare, 20);

  const proteinMin = sigmoid(0.09, 0.105 + 0.006 * pastaUsageType_k + 0.005 * dieMaterial_k + 0.007 * pastaThickness_k, 0.5, glutenShare, 20);

  const ashMax = 0.008 - 0.0015 * pastaThickness_k - 0.0005 * dieMaterial_k;

  const flourProteinMix = dot(flourProtein);
  const protein_k = sigmoid(0, 1, 0.5, 10, (flourProteinMix - proteinMin) / 0.02);
  const oilOpt = sigmoid(0, 0.03, 0.015, 15,
    0.014 + 0.006 * (pidx === 1? 1: 0) - 0.007 * protein_k - 0.003 * pastaUsageType_k - albumen * albumenFat - yolk * yolkFat
  );

  const saltOpt = sigmoid(0.003, 0.018, 0.0105, 15,
    0.0017 / (1 / 0.85 - 1) + choose([0, 0.001, -0.001], pidx) - albumen * 0.001 - yolk * 0.0008
  );

  const freshPastaWeight = 1 + additionalWater + albumen + yolk + oil + salt + barleyMalt;
  const protein_f = sigmoid(0, 0.0004, 0, 50, flourProteinMix);
  const process_f = -0.0015 * (dot(flourStrength.map((fs, i) => fs + (flourFiber[i] * 3 + (1 - flourFiber[i]) * 1.2) * flourFiber[i] + flourAsh[i] * 1.5)) / freshPastaWeight);
  const pasta_k = 0.0006 * pastaThickness_k;
  const barleyMaltAromaticOpt = (bmidx? 1: 0) * sigmoid(0.002, choose([0.005, 0.004, 0.0035], pidx), 0.0025, 15,
      (bmidx === 1? 0: 0.0025) + pasta_k + protein_f + process_f
    );

  return [
    [WMin],
    [PLOpt],
    [proteinMin],
    [ashMax],
    [oilOpt],
    [saltOpt],
    [barleyMaltAromaticOpt]
  ];
}

/**
 * FPC_LAMINATION
 * Computes rest time, lamination type, fold type, and number of passes.
 * Direct translation of your LET-based Excel model.
 *
 * @customfunction
 */
export function FPC_LAMINATION(
  fractionsRaw,
  flourMatrix,
  flourTemperature,
  airTemperature,
  airRelativeHumidity,
  albumen,
  yolk,
  oil,
  salt,
  pastaThickness,
  pastaProcess,
  dieMaterial,
  pastaUsageType,
  totalWater,
  hydrationDeficit
){

  const normalized = normalizeFractions(fractionsRaw);

  const col = index => take(flourMatrix.map(r => r[index]), flours);
  const flourStrength = col(0);
  const flourPL       = col(1);
  const flourProtein  = col(4);
  const flourFiber    = col(6);
  const flourAsh      = col(7);
  const flourType     = col(9);

  const dot = v => sumproduct(v, normalized);

  const fidx = flourType.map(t => xmatch(t, flourTypes));
  const pidx  = xmatch(pastaProcess, processes);
  const puidx = xmatch(pastaUsageType, usageTypes);
  const dmidx = xmatch(dieMaterial, dieTypes);

  const wholeFlour = flourFiber.map((fb, i) =>
    fb >= choose([0.018,0.019,0.019,0.020,0.019,0.018,0.030,0.020,0.022,0.020,0.030,0.028,0.025], fidx[i])
    || flourAsh[i] >= choose([0.008,0.0085,0.0085,0.009,0.009,0.0085,0.015,0.010,0.010,0.010,0.013,0.011,0.012], fidx[i])
  );

  const baseRestTime = dot(fidx.map(idx => choose([14,18,16,18,15,12,28,20,19,16,32,26,38], idx)));

  const shear_k = choose([
      1 + 0.05 * Math.log(1 + pastaThickness * 3),
      1.35,
      1.22
    ],
    pidx
  );

  const pentosanTotal = fidx.map(idx => choose([0.75,0.74,0.74,0.75,0.60,0.65,0.55,0.70,0.70,0.72,0.30,0.65,0.25], idx));
  const pentosanSoluble = fidx.map((idx, i) => choose([
      linear(0.28,0.22,wholeFlour[i]),
      linear(0.27,0.21,wholeFlour[i]),
      linear(0.27,0.22,wholeFlour[i]),
      linear(0.25,0.20,wholeFlour[i]),
      linear(0.38,0.32,wholeFlour[i]),
      0.30,0.32,0.25,0.28,0.26,0.30,0.40,0.35
    ], idx)
  );
  const pentosanSolubleMix = dot(pentosanSoluble.map((ps, i) => ps * pentosanTotal[i] * flourFiber[i]));
  const pentosanRest_k = 1 + 1.2 * (1 - Math.exp(-pentosanSolubleMix / 0.015)) * (0.4 + Math.log(1 + hydrationDeficit));

  const glutenWeakness_k = 1 + 0.45 * Math.tanh(0.6 * (0.6 - dot(flourPL)));
  const gluten_k = 1 + 0.25 * Math.tanh(0.5 * dot(flourProtein) * (1 - dot(flourPL)));
  const nonGlutenProtein_k = 1 + 0.25 * Math.tanh(0.6 * (albumen * albumenProtein + yolk * yolkProtein));
  const enzyme_k = 1 + 0.25 * Math.tanh(0.5 * (albumen * albumenProtein + yolk * yolkProtein));

  const hydration_k = 1 + 0.8 * Math.log(1 + hydrationDeficit);
  const proteinFiberInteraction = 1 + 0.25 * Math.pow(dot(flourProtein) * dot(flourFiber), 0.9);
  const structure_k = 1 + 0.5 * Math.tanh((dot(flourProtein) - 0.11) / 0.02) + 0.5 * Math.tanh(pentosanSolubleMix / 0.012) + 0.4 * Math.max(0, 0.6 - dot(flourPL));
  const plMemory_k = 1 + 0.35 * Math.tanh(1.5 * (dot(flourPL) - 0.6)) * shear_k;

  const airTemperature_k = expDecay(0.015, airTemperature - 22);
  const fatSalt_k = 1 - 0.35 * Math.tanh((0.1 * albumen * albumenFat + 0.5 * yolk * yolkFat + oil + 0.6 * salt) / 0.05);
  const process_k = choose([1.1, 0.85, 0.95], pidx);
  const usage_k = (pidx === 1? choose([1.15, 1], puidx): 1);
  const dieMaterialStructural_k = (pidx === 1)
      ? 1
      : choose([1.05, 0.95], dmidx);
  const interaction_k = 1 + 0.2 * dot(flourProtein) * dot(flourFiber);

  const restTime = Math.max(6, baseRestTime * shear_k * pentosanRest_k * glutenWeakness_k * gluten_k * nonGlutenProtein_k * enzyme_k * hydration_k * proteinFiberInteraction * structure_k * plMemory_k * airTemperature_k * fatSalt_k * process_k * usage_k * dieMaterialStructural_k * interaction_k);

  const rigidityIndex = (dot(flourStrength) * (0.8 + 0.4 * dot(flourProtein)) * (0.8 + 0.5 * dot(flourProtein)) * dot(flourPL) * (1 + 1.2 * dot(flourFiber))) / Math.pow(0.45 + softplus((totalWater - 0.25 * oil - 0.03 * salt) * (1 + 0.04 * (airRelativeHumidity - 0.50)), 25), 0.7);

  let laminationType = "--";
  if (pidx === 1){
    laminationType = rigidityIndex < 121? "same direction"
                    : rigidityIndex < 181? "mixed"
                    : "crossed";
  }

  let foldType = 0;
  if (pidx === 1){
    if (dot(flourFiber) > 0.025 || rigidityIndex > 160)
      foldType = 2;
    else if (rigidityIndex > 100)
      foldType = 3;
    else
      foldType = 4;
  }

  let passes = "--";
  if (pidx === 1){
    const raw = -1.5 * Math.log(pastaThickness / 6) * (1 + rigidityIndex / 250) * (1 + 0.3 * dot(flourFiber));
    passes = clamp(Math.ceil(raw), 4, 12);
  }

  return [
    [restTime],
    [laminationType],
    [foldType],
    [passes]
  ];
}

/**
 * FPC_COOKING
 * Computes cooking time and salt uptake ratio.
 * Direct translation of your LET-based Excel model.
 *
 * @customfunction
 */
export function FPC_COOKING(
  fractionsRaw,
  flourMatrix,
  albumen,
  yolk,
  oil,
  salt,
  additionalIngredient,
  pastaType,
  pastaThickness,
  pastaProcess,
  flourMoistureDB,
  strictlyBoundWaterDB,
  additionalWater,
  pastaYield
){

  const normalized = normalizeFractions(fractionsRaw);

  const col = index => take(flourMatrix.map(r => r[index]), flours);
  const flourCarbohydrate = col(2);
  const flourProtein      = col(4);
  const flourFat          = col(5);
  const flourSalt         = col(8);
  const flourType         = col(9);

  const dot = v => sumproduct(v, normalized);

  const pidx = xmatch(pastaProcess, processes);

  const flourMoistureEffective = moistureDryToWet(flourMoistureDB - strictlyBoundWaterDB);

  const pastaTotal = 1 + additionalWater + albumen + yolk + oil + salt + additionalIngredient;

  const waterLost = 1 - pastaYield;

  const pastaWater =
    (flourMoistureEffective + additionalWater + albumen * albumenWater + yolk * yolkWater + additionalIngredient * lookup(pastaType, pastaTypes, [0.012, 0.007], 0)) / pastaTotal - waterLost;

  const pastaProtein = (dot(flourProtein) + albumen * albumenProtein + yolk * yolkProtein) / pastaTotal;
  const pastaFat = (dot(flourFat) + albumen * albumenFat + yolk * yolkFat + oil) / pastaTotal;
  const pastaCarbohydrate = (dot(flourCarbohydrate) + albumen * 0.009 + yolk * 0.005) / pastaTotal;
  const pastaSalt = (dot(flourSalt) + albumen * 0.0042 + yolk * 0.001 + salt) / pastaTotal;

  const cookTimeStarch = 90 + 270 * pastaCarbohydrate;

  const structure_k = Math.exp(-1.5 * pastaProtein - 0.8 * pastaFat);
  const hydration_k = 0.6 + 0.8 * pastaWater;
  const opening_k = expRise(1, 1.6, 1 / 600, cookTimeStarch);
  const drying_k = sigmoid(1, 1.35, 0.95, 50, pastaYield);

  const Ea = choose([28000, 24000, 26000], pidx);
  const T = 99 - 0.40 * pastaThickness;
  const diffusivityEffective = 0.0000012 * arrhenius(Ea, T) * structure_k * hydration_k * opening_k * drying_k;

  const L = pastaThickness / 1000;
  const cookTimeDiffusive = diffusionTime(L, diffusivityEffective, 0.7);

  const cookTimeThermal = 480 * arrhenius(Ea, T, 99);

  const cookTime = Math.max(cookTimeStarch, cookTimeDiffusive, cookTimeThermal);

  const targetSalt = 0.0055;
  const XSalt = diffusionFraction(L, 4 * diffusivityEffective, cookTime);
  const cookSaltRatio = Math.max(0, (targetSalt - pastaSalt) / XSalt);

  return [
    [cookTime / 60],   // minutes
    [cookSaltRatio]
  ];
}
