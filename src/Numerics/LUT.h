#pragma once
#include "Leibniz.h"

// Leibniz HPC Math Library
// Taylor/Horner LUT coefficient tables for all Scalar* transcendental functions.
//
// TODO: The following Balanced/Approximate coefficient arrays contain incorrect c[0] values
//       (max error values were accidentally placed as the leading coefficient).
//       Re-run scripts/remez_gen.py and paste the output to fix:
//         - s_Expm1BalancedCoefficients      [c[0] should be ~1.0]
//         - s_Expm1ApproximateCoefficients   [c[0] should be ~1.0]
//         - s_Log1pBalancedCoefficients      [c[0] should be ~1.0]
//         - s_Log1pApproximateCoefficients   [c[0] should be ~1.0]
//       Note: Fix the COS section in remez_gen.py first (uses cosh instead of cos,
//       and domain 1.0 instead of quarter_pi) before regenerating.
//
// Layout rules:
//   - All arrays are std::array<double, MAX_TERMS> (20 entries) for uniform class layout.
//   - Trailing entries are 0.0 where the series has fewer than 20 meaningful terms.
//   - Three tiers per function: Exact (20 terms), Balanced (10 terms), Approximate (8 terms).
//     Balanced and Approximate entries are zeroed - caller fills in tuned Remez coefficients.
//   - Functions whose evaluation reduces to other primitives (Exp, Log, Atan, Sqrt, Erf)
//     are marked with a comment explaining the omission. No redundant tables are stored.
//
// Evaluation conventions:
//   - Straight series  (Exp, Log, Expm1, Log1p, Log2, Log10, Tan):
//       result = Horner(x, c)
//   - Odd series       (Sin, Atan, Sinh, Erf, Asin):
//       result = x * Horner(x^2, c)          [c[0] is the x^1 coefficient]
//   - Even series      (Cos, Cosh):
//       result = Horner(x^2, c)              [c[0] is the x^0 coefficient]
//   - Exp2 / Exp10:
//       c[n] = (ln2)^n / n!  and  (ln10)^n / n!  respectively
//   - Log2 / Log10:
//       c[n] = kLogP1[n] / ln2  and  kLogP1[n] / ln10  respectively
//   - Seed polynomials (Sqrt, Rsqrt, Cbrt):
//       short minimax seeds, Newton-Raphson refinement handles accuracy
//   - Lgamma / Tgamma:
//       Lanczos approximation coefficients, g = 7, 9-term rational sum

namespace Leibniz::Numerics::Lookup {
	class LEIBNIZ_RUNTIME_API LookUpTables {
		constexpr static size_t MAX_TERMS = 20;

	public:

		// EXP
		// Taylor series of e^x at x = 0.
		// Argument reduction: x = k*ln2 + r, evaluate on r, scale by 2^k.
		// c[n] = 1 / n!
		// Straight Horner on x.

		inline static std::array<double, MAX_TERMS> s_ExpExactCoefficients = {
			1.0,                        // 1/0!
			1.0,                        // 1/1!
			0.5,                        // 1/2!
			0.16666666666666666667,     // 1/3!
			0.04166666666666666667,     // 1/4!
			0.00833333333333333333,     // 1/5!
			0.00138888888888888889,     // 1/6!
			0.00019841269841269841,     // 1/7!
			0.00002480158730158730,     // 1/8!
			0.00000275573192239859,     // 1/9!
			0.00000027557319223986,     // 1/10!
			0.00000002505210838544,     // 1/11!
			0.00000000208767569879,     // 1/12!
			0.00000000016059043836,     // 1/13!
			0.00000000001147074560,     // 1/14!
			0.00000000000076471637,     // 1/15!
			0.00000000000004779477,     // 1/16!
			0.00000000000000281146,     // 1/17!
			0.00000000000000015619,     // 1/18!
			0.00000000000000000822      // 1/19!
		};

		// Balanced max error: 1.3493e-14
		inline static std::array<double, MAX_TERMS> s_ExpBalancedCoefficients = {
			1.0000000000000135,
			1.0000000000000123,
			0.49999999999438426,
			0.16666666666493138,
			0.041666667040619957,
			0.0083333334040310269,
			0.0013888801740591498,
			0.00019841149885492403,
			2.4884463525369608e-5,
			2.7647723819294138e-6,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 4.04652e-11
		inline static std::array<double, MAX_TERMS> s_ExpApproximateCoefficients = {
			0.99999999995953477,
			0.99999999996405271,
			0.50000001077696019,
			0.16666666995840139,
			0.041666218229511866,
			0.0083332486476706380,
			0.0013948583309728026,
			0.00019924169071930781,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// EXPM1
		// Taylor series of (e^x - 1) at x = 0.
		// Identical to Exp but the constant term (1/0! = 1) is dropped.
		// This avoids catastrophic cancellation for small x.
		// c[n] = 1 / (n+1)!   stored starting from the x^1 term.
		// Straight Horner on x, then multiply result by x.

		inline static std::array<double, MAX_TERMS> s_Expm1ExactCoefficients = {
			1.0,                        // 1/1!
			0.5,                        // 1/2!
			0.16666666666666666667,     // 1/3!
			0.04166666666666666667,     // 1/4!
			0.00833333333333333333,     // 1/5!
			0.00138888888888888889,     // 1/6!
			0.00019841269841269841,     // 1/7!
			0.00002480158730158730,     // 1/8!
			0.00000275573192239859,     // 1/9!
			0.00000027557319223986,     // 1/10!
			0.00000002505210838544,     // 1/11!
			0.00000000208767569879,     // 1/12!
			0.00000000016059043836,     // 1/13!
			0.00000000001147074560,     // 1/14!
			0.00000000000076471637,     // 1/15!
			0.00000000000004779477,     // 1/16!
			0.00000000000000281146,     // 1/17!
			0.00000000000000015619,     // 1/18!
			0.00000000000000000822,     // 1/19!
			0.00000000000000000041      // 1/20!
		};

		// Balanced max error: 1.3493e-14
		inline static std::array<double, MAX_TERMS> s_Expm1BalancedCoefficients = {
			1.3492955327249727e-14,
			1.0000000000000123,
			0.49999999999438426,
			0.16666666666493138,
			0.041666667040619957,
			0.0083333334040310269,
			0.0013888801740591498,
			0.00019841149885492403,
			2.4884463525369608e-5,
			2.7647723819294138e-6,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 4.04652e-11
		inline static std::array<double, MAX_TERMS> s_Expm1ApproximateCoefficients = {
			-4.0465229035249177e-11,
			0.99999999996405271,
			0.50000001077696019,
			0.16666666995840139,
			0.041666218229511866,
			0.0083332486476706380,
			0.0013948583309728026,
			0.00019924169071930781,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// EXP2
		// Taylor series of 2^x = e^(x*ln2) at x = 0.
		// c[n] = (ln2)^n / n!
		// ln2 = 0.69314718055994530942
		// Straight Horner on x.

		inline static std::array<double, MAX_TERMS> s_Exp2ExactCoefficients = {
			1.0,                        // (ln2)^0 / 0!
			0.69314718055994530942,     // (ln2)^1 / 1!
			0.24022650695910071233,     // (ln2)^2 / 2!
			0.05550410866482157453,     // (ln2)^3 / 3!
			0.00961812910762847716,     // (ln2)^4 / 4!
			0.00133335581463453406,     // (ln2)^5 / 5!
			0.00015403530393381610,     // (ln2)^6 / 6!
			0.00001525273380405984,     // (ln2)^7 / 7!
			0.00000132730627751550,     // (ln2)^8 / 8!
			0.00000010238929750648,     // (ln2)^9 / 9!
			0.00000000709892634689,     // (ln2)^10 / 10!
			0.00000000044728108678,     // (ln2)^11 / 11!
			0.00000000002583771004,     // (ln2)^12 / 12!
			0.00000000000137858809,     // (ln2)^13 / 13!
			0.00000000000006815539,     // (ln2)^14 / 14!
			0.00000000000000314633,     // (ln2)^15 / 15!
			0.00000000000000013624,     // (ln2)^16 / 16!
			0.00000000000000000555,     // (ln2)^17 / 17!
			0.00000000000000000021,     // (ln2)^18 / 18!
			0.00000000000000000001      // (ln2)^19 / 19!
		};

		// Balanced max error: 1.3493e-14
		inline static std::array<double, MAX_TERMS> s_Exp2BalancedCoefficients = {
			1.0000000000000135,
			0.69314718055995381,
			0.24022650695640261,
			0.055504108664243688,
			0.0096181291939500217,
			0.0013333558259546660,
			0.00015403433741197383,
			1.5252641589521348e-5,
			1.3259647255653790e-6,
			1.0211476258095146e-7,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 4.04652e-11
		inline static std::array<double, MAX_TERMS> s_Exp2ApproximateCoefficients = {
			0.99999999995953477,
			0.69314718053502855,
			0.24022651213692372,
			0.055504109761050390,
			0.0096180255925936403,
			0.0013333422647083517,
			0.00015469734740832799,
			1.5316461574910607e-5,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// EXP10
		// Taylor series of 10^x = e^(x*ln10) at x = 0.
		// c[n] = (ln10)^n / n!
		// ln10 = 2.30258509299404568402
		// Straight Horner on x.

		inline static std::array<double, MAX_TERMS> s_Exp10ExactCoefficients = {
			1.0,                        // (ln10)^0 / 0!
			2.30258509299404568402,     // (ln10)^1 / 1!
			2.65094905530562491363,     // (ln10)^2 / 2!
			2.03470972671920792206,     // (ln10)^3 / 3!
			1.17125663593045499320,     // (ln10)^4 / 4!
			0.54008677703546359498,     // (ln10)^5 / 5!
			0.20748831461036313977,     // (ln10)^6 / 6!
			0.06830831745243545240,     // (ln10)^7 / 7!
			0.01966823751989462588,     // (ln10)^8 / 8!
			0.00503398691413771400,     // (ln10)^9 / 9!
			0.00115913535981994278,     // (ln10)^10 / 10!
			0.00024277580892576050,     // (ln10)^11 / 11!
			0.00004661922082481098,     // (ln10)^12 / 12!
			0.00000825539867043413,     // (ln10)^13 / 13!
			0.00000135827728032095,     // (ln10)^14 / 14!
			0.00000020861975523405,     // (ln10)^15 / 15!
			0.00000003002537003163,     // (ln10)^16 / 16!
			0.00000000406778531543,     // (ln10)^17 / 17!
			0.00000000052020614073,     // (ln10)^18 / 18!
			0.00000000006305390028      // (ln10)^19 / 19!
		};

		// Balanced max error: 2.26929e-9
		inline static std::array<double, MAX_TERMS> s_Exp10BalancedCoefficients = {
			1.0000000022692923,
			2.3025850977223830,
			2.6509486022847377,
			2.0346782709261900,
			1.1712696109374268,
			0.53938921270589086,
			0.20683438744294122,
			0.068038254066088306,
			0.020331518836359685,
			0.0051981270297811038,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 6.20463e-7
		inline static std::array<double, MAX_TERMS> s_Exp10ApproximateCoefficients = {
			0.99999937953695601,
			2.3025838314934113,
			2.6510281854871222,
			2.0347340380543156,
			1.1696794689455906,
			0.53869932256456735,
			0.21700479435063549,
			0.071286110993006912,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// LOG1P  (log(1+u))
		// Taylor series of ln(1+u) at u = 0.
		// Argument reduction maps x into [0.5, 1.5] via IEEE exponent extraction,
		// then u = x - 1 lands in (-0.5, 0.5] for convergence.
		// c[n] = (-1)^(n+1) / n
		// Straight Horner on u.

		inline static std::array<double, MAX_TERMS> s_Log1pExactCoefficients = {
			 1.0,                       //  1/1
			-0.5,                       // -1/2
			 0.33333333333333333333,    //  1/3
			-0.25,                      // -1/4
			 0.2,                       //  1/5
			-0.16666666666666666667,    // -1/6
			 0.14285714285714285714,    //  1/7
			-0.125,                     // -1/8
			 0.11111111111111111111,    //  1/9
			-0.1,                       // -1/10
			 0.09090909090909090909,    //  1/11
			-0.08333333333333333333,    // -1/12
			 0.07692307692307692308,    //  1/13
			-0.07142857142857142857,    // -1/14
			 0.06666666666666666667,    //  1/15
			-0.0625,                    // -1/16
			 0.05882352941176470588,    //  1/17
			-0.05555555555555555556,    // -1/18
			 0.05263157894736842105,    //  1/19
			-0.05                       // -1/20
		};

		// Balanced max error: 3.81553e-7
		inline static std::array<double, MAX_TERMS> s_Log1pBalancedCoefficients = {
			-3.8155269052064877e-7,
			1.0000035050620309,
			-0.49992693149864173,
			0.33309800534958234,
			-0.25222429891950878,
			0.20446348299568440,
			-0.14344349205218564,
			0.10864928168906704,
			-0.21842355201645804,
			0.21956340609568409,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 6.64293e-6
		inline static std::array<double, MAX_TERMS> s_Log1pApproximateCoefficients = {
			6.6429292723428828e-6,
			0.99995217386199359,
			-0.50080680268150843,
			0.33539779424413027,
			-0.23487091617709928,
			0.17570346724024897,
			-0.25343381084445765,
			0.24506877072435033,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// LOG2
		// log2(1+u) = ln(1+u) / ln2.
		// Same series as Log1p with every coefficient divided by ln2.
		// ln2 = 0.69314718055994530942
		// Straight Horner on u.

		inline static std::array<double, MAX_TERMS> s_Log2ExactCoefficients = {
			 1.44269504088896340736,    //  1 / ln2
			-0.72134752044448170368,    // -1/(2*ln2)
			 0.48089834696298780245,    //  1/(3*ln2)
			-0.36067376022224085184,    // -1/(4*ln2)
			 0.28853900817779268147,    //  1/(5*ln2)
			-0.24044917348149390123,    // -1/(6*ln2)
			 0.20609929155556620105,    //  1/(7*ln2)
			-0.18033688011112042592,    // -1/(8*ln2)
			 0.16029944898766260082,    //  1/(9*ln2)
			-0.14426950408889634074,    // -1/(10*ln2)
			 0.13115409462626030067,    //  1/(11*ln2)
			-0.12022458674074695061,    // -1/(12*ln2)
			 0.11097654160684334057,    //  1/(13*ln2)
			-0.10304964577778310053,    // -1/(14*ln2)
			 0.09617966939259756049,    //  1/(15*ln2)
			-0.09016843755555896296,    // -1/(16*ln2)
			 0.08486441228758608279,    //  1/(17*ln2)
			-0.08014972449383130041,    // -1/(18*ln2)
			 0.07593132836258754776,    //  1/(19*ln2)
			-0.07213475204444817037     // -1/(20*ln2)
		};

		// Balanced max error: 5.50464e-7
		inline static std::array<double, MAX_TERMS> s_Log2BalancedCoefficients = {
			-5.5046417445198138e-7,
			1.4427000976245734,
			-0.72124210487992695,
			0.48055884044784784,
			-0.36388274524287085,
			0.29497845296075878,
			-0.20694521463148366,
			0.15674777988896508,
			-0.31511857530749655,
			0.31676303713493304,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 9.58372e-6
		inline static std::array<double, MAX_TERMS> s_Log2ApproximateCoefficients = {
			9.5837211181852072e-6,
			1.4426260423568367,
			-0.72251149067206984,
			0.48387673448110366,
			-0.33884710601774854,
			0.25348652085450363,
			-0.36562770209889065,
			0.35355950020077460,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// LOG10
		// log10(1+u) = ln(1+u) / ln10.
		// Same series as Log1p with every coefficient divided by ln10.
		// ln10 = 2.30258509299404568402
		// Straight Horner on u.

		inline static std::array<double, MAX_TERMS> s_Log10ExactCoefficients = {
			 0.43429448190325182765,    //  1 / ln10
			-0.21714724095162591383,    // -1/(2*ln10)
			 0.14476482730108394255,    //  1/(3*ln10)
			-0.10857362047581295691,    // -1/(4*ln10)
			 0.08685889638065036553,    //  1/(5*ln10)
			-0.07238241365054197128,    // -1/(6*ln10)
			 0.06204207027189311824,    //  1/(7*ln10)
			-0.05428681023790647846,    // -1/(8*ln10)
			 0.04825494243369464752,    //  1/(9*ln10)
			-0.04342944819032518277,    // -1/(10*ln10)
			 0.03948131653665925706,    //  1/(11*ln10)
			-0.03619120682527098564,    // -1/(12*ln10)
			 0.03340726299255014059,    //  1/(13*ln10)
			-0.03102103513594655912,    // -1/(14*ln10)
			 0.02895296546021678851,    //  1/(15*ln10)
			-0.02714340511895323923,    // -1/(16*ln10)
			 0.02554085188842069575,    //  1/(17*ln10)
			-0.02412747121684732376,    // -1/(18*ln10)
			 0.02285760431069746461,    //  1/(19*ln10)
			-0.02171472409516259138     // -1/(20*ln10)
		};

		// Balanced max error: 1.65706e-7
		inline static std::array<double, MAX_TERMS> s_Log10BalancedCoefficients = {
			-1.6570622804845694e-7,
			0.43429600413235057,
			-0.21711550770468508,
			0.14466262565630347,
			-0.10953962122265899,
			0.088797362415745096,
			-0.062296717063197183,
			0.047185783500313838,
			-0.094860143358455620,
			0.095355175695238406,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 2.88499e-6
		inline static std::array<double, MAX_TERMS> s_Log10ApproximateCoefficients = {
			2.8849875266520979e-6,
			0.43427371127542490,
			-0.21749763090418977,
			0.14566141128274802,
			-0.10200314285527542,
			0.076307046273708906,
			-0.11006490557746046,
			0.10643201481239854,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// SIN
		// Taylor series of sin(x) at x = 0, odd terms only.
		// Argument reduction to [-pi/4, pi/4] required before evaluation.
		// c[n] = (-1)^n / (2n+1)!
		// Evaluate as: x * Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_SineExactCoefficients = {
			 1.0,                       //  1/1!
			-0.16666666666666666667,    // -1/3!
			 0.00833333333333333333,    //  1/5!
			-0.00019841269841269841,    // -1/7!
			 0.00000275573192239859,    //  1/9!
			-0.00000002505210838544,    // -1/11!
			 0.00000000016059043836,    //  1/13!
			-0.00000000000076471637,    // -1/15!
			 0.00000000000000281146,    //  1/17!
			-0.00000000000000000822,    // -1/19!
			 0.00000000000000000002,    //  1/21! - below double epsilon, kept for uniformity
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced max error: 2.95782e-28
		inline static std::array<double, MAX_TERMS> s_SineBalancedCoefficients = {
			1.0000000000000000,
			-0.16666666666666667,
			0.0083333333333333333,
			-0.00019841269841269841,
			2.7557319223985879e-6,
			-2.5052108385434924e-8,
			1.6059043834317589e-10,
			-7.6471631606173216e-13,
			2.8113784967206628e-15,
			-8.1604803118106633e-18,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 1.784e-21
		inline static std::array<double, MAX_TERMS> s_SineApproximateCoefficients = {
			1.0000000000000000,
			-0.16666666666666667,
			0.0083333333333333207,
			-0.00019841269841253485,
			2.7557319213565214e-6,
			-2.5052104779805832e-8,
			1.6058352512671332e-10,
			-7.5780940422941733e-13,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// COS
		// Taylor series of cos(x) at x = 0, even terms only.
		// Shares argument reduction with Sin.
		// c[n] = (-1)^n / (2n)!
		// Evaluate as: Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_CosineExactCoefficients = {
			 1.0,                       //  1/0!
			-0.5,                       // -1/2!
			 0.04166666666666666667,    //  1/4!
			-0.00138888888888888889,    // -1/6!
			 0.00002480158730158730,    //  1/8!
			-0.00000027557319223986,    // -1/10!
			 0.00000000208767569879,    //  1/12!
			-0.00000000001147074560,    // -1/14!
			 0.00000000000004779477,    //  1/16!
			-0.00000000000000015619,    // -1/18!
			 0.00000000000000000042,    //  1/20! - below double epsilon, kept for uniformity
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced max error: 7.93367e-25
		inline static std::array<double, MAX_TERMS> s_CosineBalancedCoefficients = {
			1.0000000000000000,
			0.50000000000000000,
			0.041666666666666667,
			0.0013888888888888890,
			2.4801587301586866e-5,
			2.7557319224148321e-7,
			2.0876756950968977e-9,
			1.1470750784498902e-11,
			4.7790369302047721e-14,
			1.5826018056891801e-16,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 1.48018e-18
		inline static std::array<double, MAX_TERMS> s_CosineApproximateCoefficients = {
			1.0000000000000000,
			0.50000000000000019,
			0.041666666666662691,
			0.0013888888889206766,
			2.4801587176784215e-5,
			2.7557345825742232e-7,
			2.0873617441255408e-9,
			1.1663435515941556e-11,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// TAN
		// Taylor series of tan(x) at x = 0 via Bernoulli numbers.
		// Valid only for |x| < pi/2. Argument reduction to (-pi/4, pi/4) required.
		// c[n] = (-1)^(n-1) * 4^n * (4^n - 1) * B(2n) / (2n)!
		// where B(2n) are Bernoulli numbers: B2=1/6, B4=-1/30, B6=1/42, ...
		// Straight Horner on x (odd powers only; evaluate as x * Horner(x^2, c)).

		inline static std::array<double, MAX_TERMS> s_TanExactCoefficients = {
			 1.0,                       // x^1  coefficient
			 0.33333333333333333333,    // x^3  : 1/3
			 0.13333333333333333333,    // x^5  : 2/15
			 0.05396825396825396825,    // x^7  : 17/315
			 0.02186948853615520282,    // x^9  : 62/2835
			 0.00886323552990219656,    // x^11 : 1382/155925
			 0.00359212803657248101,    // x^13 : 21844/6081075
			 0.00145583438705131827,    // x^15 : 929569/638512875
			 0.00059002744995575946,    // x^17
			 0.00023912675844522943,    // x^19
			 0.00009691537956929451,    // x^21
			 0.00003927832388331683,    // x^23
			 0.00001591589731760018,    // x^25
			 0.00000644972236960897,    // x^27
			 0.00000261244890210170,    // x^29
			 0.00000105824056930732,    // x^31
			 0.00000042882905518350,    // x^33
			 0.00000017377501751165,    // x^35
			 0.00000007040656870072,    // x^37
			 0.00000002851952512993     // x^39
		};

		// Balanced max error: 6.813e-12
		inline static std::array<double, MAX_TERMS> s_TanBalancedCoefficients = {
			0.99999999999318700,
			0.33333333548707447,
			0.13333322113488597,
			0.053970515082666067,
			0.021846453550244046,
			0.0089970228949667362,
			0.0031248912011567273,
			0.0024438424067947911,
			-0.00061770125045039413,
			0.00096684922248177998,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 1.32169e-9
		inline static std::array<double, MAX_TERMS> s_TanApproximateCoefficients = {
			0.99999999867831231,
			0.33333359902061648,
			0.13332459049323400,
			0.054077286625233495,
			0.021208210475267699,
			0.011001889244338787,
			-9.8266822469273380e-5,
			0.0044605507908773809,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// ASIN
		// Taylor series of asin(x) at x = 0, odd terms only.
		// Valid for |x| <= 1. Near |x| = 1 use: asin(x) = pi/2 - 2*asin(sqrt ((1-x)/2)).
		// c[n] = (2n)! / (4^n * (n!)^2 * (2n+1))   [Wallis product coefficients]
		// Evaluate as: x * Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_AsinExactCoefficients = {
			 1.0,                       // x^1  : 1
			 0.16666666666666666667,    // x^3  : 1/6
			 0.07500000000000000000,    // x^5  : 3/40
			 0.04464285714285714286,    // x^7  : 15/336
			 0.03038194444444444444,    // x^9  : 105/3456
			 0.02237215909090909091,    // x^11 : 945/42240
			 0.01735276442307692308,    // x^13 : 10395/599040
			 0.01396484375000000000,    // x^15 : 135135/9676800
			 0.01155180089613970588,    // x^17 : 2027025/175472640
			 0.00976160952919497378,    // x^19
			 0.00836898709541088987,    // x^21
			 0.00726892733371813768,    // x^23
			 0.00637685321280280709,    // x^25
			 0.00564064580912797451,    // x^27
			 0.00502470093352093352,    // x^29
			 0.00450421367862931755,    // x^31
			 0.00405802828595578782,    // x^33
			 0.00367481226172263573,    // x^35
			 0.00334199273686328681,    // x^37
			 0.00304966876971870082     // x^39
		};

		// Balanced max error: 5.75274e-14
		inline static std::array<double, MAX_TERMS> s_AsinBalancedCoefficients = {
			0.99999999999994247,
			0.16666666671167053,
			0.074999994197036012,
			0.044643146703948404,
			0.030374636496662151,
			0.022477399624555438,
			0.016440189028243989,
			0.018767857394105430,
			-0.0031333942953440337,
			0.032191125868765618,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 1.54046e-11
		inline static std::array<double, MAX_TERMS> s_AsinApproximateCoefficients = {
			0.99999999998459540,
			0.16666667434180944,
			0.074999373760292797,
			0.044662237314864275,
			0.030089863700043181,
			0.024725496640533203,
			0.0071827605450616800,
			0.034998952629384186,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// ACOS
		// acos(x) = pi/2 - asin(x).
		// No separate coefficient table - reuses s_Asin*Coefficients at eval time.

		// ATAN
		// Taylor series of atan(x) at x = 0, odd terms only.
		// Valid for |x| <= 1. For |x| > 1: atan(x) = pi/2 - atan(1/x).
		// c[n] = (-1)^n / (2n+1)
		// Evaluate as: x * Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_AtanExactCoefficients = {
			 1.0,                       //  1/1
			-0.33333333333333333333,    // -1/3
			 0.2,                       //  1/5
			-0.14285714285714285714,    // -1/7
			 0.11111111111111111111,    //  1/9
			-0.09090909090909090909,    // -1/11
			 0.07692307692307692308,    //  1/13
			-0.06666666666666666667,    // -1/15
			 0.05882352941176470588,    //  1/17
			-0.05263157894736842105,    // -1/19
			 0.04761904761904761905,    //  1/21
			-0.04347826086956521739,    // -1/23
			 0.04,                      //  1/25
			-0.03703703703703703704,    // -1/27
			 0.03448275862068965517,    //  1/29
			-0.03225806451612903226,    // -1/31
			 0.03030303030303030303,    //  1/33
			-0.02857142857142857143,    // -1/35
			 0.02702702702702702703,    //  1/37
			-0.02564102564102564103     // -1/39
		};

		// Balanced max error: 2.06851e-9
		inline static std::array<double, MAX_TERMS> s_AtanBalancedCoefficients = {
			0.99999999793149424,
			-0.33333288299041320,
			0.19998385999608258,
			-0.14263131277780034,
			0.10948056429099576,
			-0.083944029281457054,
			0.057837559773673265,
			-0.031083153031170269,
			0.010873789642375190,
			-0.0017862322248376252,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 8.64888e-8
		inline static std::array<double, MAX_TERMS> s_AtanApproximateCoefficients = {
			0.99999991351122136,
			-0.33332106793021934,
			0.19971508960011209,
			-0.14029879899536988,
			0.099433283340317761,
			-0.059904448435683837,
			0.024551860447480324,
			-0.0047777546291888000,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// ATAN2
		// atan2(y, x) is computed from Atan with quadrant correction logic.
		// No separate coefficient table - reuses s_Atan*Coefficients at eval time.

		// SINH
		// Taylor series of sinh(x) at x = 0, odd terms only.
		// All positive (no alternating signs unlike Sin).
		// For |x| > 20: use (e^x - e^-x) / 2 directly to avoid cancellation.
		// c[n] = 1 / (2n+1)!
		// Evaluate as: x * Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_SinhExactCoefficients = {
			 1.0,                       //  1/1!
			 0.16666666666666666667,    //  1/3!
			 0.00833333333333333333,    //  1/5!
			 0.00019841269841269841,    //  1/7!
			 0.00000275573192239859,    //  1/9!
			 0.00000002505210838544,    //  1/11!
			 0.00000000016059043836,    //  1/13!
			 0.00000000000076471637,    //  1/15!
			 0.00000000000000281146,    //  1/17!
			 0.00000000000000000822,    //  1/19!
			 0.00000000000000000002,    //  1/21! - below double epsilon, kept for uniformity
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced max error: 3.77403e-26
		inline static std::array<double, MAX_TERMS> s_SinhBalancedCoefficients = {
			1.0000000000000000,
			0.16666666666666667,
			0.0083333333333333333,
			0.00019841269841269842,
			2.7557319223985684e-6,
			2.5052108385518996e-8,
			1.6059043819266068e-10,
			7.6471661996780065e-13,
			2.8112476920179676e-15,
			8.3190629051310749e-18,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 8.69349e-20
		inline static std::array<double, MAX_TERMS> s_SinhApproximateCoefficients = {
			1.0000000000000000,
			0.16666666666666668,
			0.0083333333333330998,
			0.00019841269841456565,
			2.7557319150670776e-6,
			2.5052124013957305e-8,
			1.6057199090236082e-10,
			7.7604168171618178e-13,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// COSH
		// Taylor series of cosh(x) at x = 0, even terms only.
		// All positive. Same bailout threshold as Sinh for large |x|.
		// c[n] = 1 / (2n)!
		// Evaluate as: Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_CoshExactCoefficients = {
			 1.0,                       //  1/0!
			 0.5,                       //  1/2!
			 0.04166666666666666667,    //  1/4!
			 0.00138888888888888889,    //  1/6!
			 0.00002480158730158730,    //  1/8!
			 0.00000027557319223986,    //  1/10!
			 0.00000000208767569879,    //  1/12!
			 0.00000000001147074560,    //  1/14!
			 0.00000000000004779477,    //  1/16!
			 0.00000000000000015619,    //  1/18!
			 0.00000000000000000042,    //  1/20! - below double epsilon, kept for uniformity
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced max error: 7.93367e-25
		inline static std::array<double, MAX_TERMS> s_CoshBalancedCoefficients = {
			1.0000000000000000,
			0.50000000000000000,
			0.041666666666666667,
			0.0013888888888888890,
			2.4801587301586866e-5,
			2.7557319224148321e-7,
			2.0876756950968977e-9,
			1.1470750784498902e-11,
			4.7790369302047721e-14,
			1.5826018056891801e-16,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 1.48018e-18
		inline static std::array<double, MAX_TERMS> s_CoshApproximateCoefficients = {
			1.0000000000000000,
			0.50000000000000019,
			0.041666666666662691,
			0.0013888888889206766,
			2.4801587176784215e-5,
			2.7557345825742232e-7,
			2.0873617441255408e-9,
			1.1663435515941556e-11,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// TANH
		// tanh(x) = sinh(x) / cosh(x).
		// No separate coefficient table - computed from s_Sinh* and s_Cosh* at eval time.

		// SQRT
		// Not a Taylor series. Uses a minimax seed polynomial over [0.5, 2.0)
		// followed by 1-2 Newton-Raphson iterations: x_{n+1} = (x_n + a/x_n) / 2.
		// IEEE exponent extraction reduces input to [0.5, 2.0) before seed eval.
		// Seed polynomial degree is 4 (5 terms). Remaining entries are 0.0.

		inline static std::array<double, MAX_TERMS> s_SqrtSeedExactCoefficients = {
			 0.17507985753745012766,    // minimax seed p0
			 1.26268988687865788626,    // minimax seed p1
			-0.83699202458812122527,    // minimax seed p2
			 0.46802808918308036404,    // minimax seed p3
			-0.07879957940947228718,    // minimax seed p4
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced max error: 0.00022853
		inline static std::array<double, MAX_TERMS> s_SqrtSeedBalancedCoefficients = {
			0.27260216044272289,
			1.0798070749978679,
			-0.50233560117824084,
			0.17644785127592012,
			-0.026279480244656835,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 0.000982389
		inline static std::array<double, MAX_TERMS> s_SqrtSeedApproximateCoefficients = {
			0.31154304356146250,
			0.92059096123301853,
			-0.27853580442911453,
			0.047076775358247518,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// RSQRT
		// Not a Taylor series. Minimax seed polynomial over [0.5, 2.0),
		// followed by 1-2 Newton-Raphson iterations: x_{n+1} = x_n * (1.5 - 0.5*a*x_n^2).
		// Seed polynomial degree is 4 (5 terms). Remaining entries are 0.0.

		inline static std::array<double, MAX_TERMS> s_RsqrtSeedExactCoefficients = {
			 1.77627265330128064765,    // minimax seed p0
			-1.40548406985049399516,    // minimax seed p1
			 0.97535974700047022183,    // minimax seed p2
			-0.37208153342741210767,    // minimax seed p3
			 0.05629370826748617694,    // minimax seed p4
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		inline static std::array<double, MAX_TERMS> s_RsqrtSeedBalancedCoefficients = {
			2.5040551589829487,
			-3.2951938595876005,
			2.7495704858546411,
			-1.1458414142374613,
			0.18524399068388448,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 0.00671091
		inline static std::array<double, MAX_TERMS> s_RsqrtSeedApproximateCoefficients = {
			2.2349007327122814,
			-2.1875804712393409,
			1.1840368469257476,
			-0.23693641273639236,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// CBRT
		// Not a Taylor series. Minimax seed polynomial over [0.5, 1.0),
		// followed by Newton-Raphson iterations: x_{n+1} = x_n*(2/3) + a/(3*x_n^2).
		// IEEE exponent extraction (mod 3) reduces input before seed eval.
		// Seed polynomial degree is 4 (5 terms). Remaining entries are 0.0.

		inline static std::array<double, MAX_TERMS> s_CbrtSeedExactCoefficients = {
			 0.61202001372078029694,    // minimax seed p0
			 0.86389031032291174775,    // minimax seed p1
			-0.56980263724990773765,    // minimax seed p2
			 0.27430536394745040498,    // minimax seed p3
			-0.05895143985793026823,    // minimax seed p4
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		inline static std::array<double, MAX_TERMS> s_CbrtSeedBalancedCoefficients = {
			0.40292687942035095,
			1.1369250845060658,
			-0.94792123059726618,
			0.54072953717515926,
			-0.13266828300345631,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 6.37106e-5
		inline static std::array<double, MAX_TERMS> s_CbrtSeedApproximateCoefficients = {
			0.43950777932264562,
			0.92883885203727841,
			-0.51302082962447160,
			0.14473790881784543,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// POW
		// pow(x, y) = exp(y * log(x)).
		// No separate coefficient table - reuses s_Exp* and s_Log1p* at eval time.

		// HYPOT
		// hypot(a, b) = sqrt(a^2 + b^2) with overflow-safe prescaling.
		// No separate coefficient table - reuses s_SqrtSeed* at eval time.

		// SIGMOID
		// sigmoid(x) = 1 / (1 + exp(-x)).
		// No separate coefficient table - reuses s_Exp* at eval time.
		// Clamps to 0 or 1 for large |x| to avoid overflow.

		// SOFTPLUS
		// softplus(x) = log(1 + exp(x)).
		// No separate coefficient table - reuses s_Exp* and s_Log1p* at eval time.

		// ERF
		// Taylor series of erf(x) at x = 0, odd terms only.
		// c[n] = (2/sqrt(pi)) * (-1)^n / (n! * (2n+1))
		// 2/sqrt(pi) = 1.12837916709551257390
		// Valid for |x| < 4. For |x| >= 4 switch to erfc asymptotic expansion.
		// Evaluate as: x * Horner(x^2, c)

		inline static std::array<double, MAX_TERMS> s_ErfExactCoefficients = {
			 1.12837916709551257390,    //  (2/sqrt pi) * 1/1
			-0.37612638903183752463,    //  (2/sqrt pi) * -1/3
			 0.11283791670955125739,    //  (2/sqrt pi) * 1/10
			-0.02686617064513124010,    //  (2/sqrt pi) * -1/42
			 0.00522397762544218268,    //  (2/sqrt pi) * 1/216
			-0.00085971308279991350,    //  (2/sqrt pi) * -1/1320
			 0.00012300158157076285,    //  (2/sqrt pi) * 1/9360
			-0.00001561989482653430,    //  (2/sqrt pi) * -1/75600
			 0.00000178520560044788,    //  n=8
			-0.00000018548459958795,    //  n=9
			 0.00000001760553010858,    //  n=10
			-0.00000000153025527940,    //  n=11
			 0.00000000012268765828,    //  n=12
			-0.00000000000910944025,    //  n=13
			 0.00000000000062873961,    //  n=14
			-0.00000000000004045263,    //  n=15
			 0.00000000000000243608,    //  n=16
			-0.00000000000000013758,    //  n=17
			 0.00000000000000000731,    //  n=18
			-0.00000000000000000037     //  n=19
		};

		// Balanced max error: 1.79907e-14
		inline static std::array<double, MAX_TERMS> s_ErfBalancedCoefficients = {
			1.1283791670954946,
			-0.37612638902820846,
			0.11283791658873862,
			-0.026866169083893008,
			0.0052239673643921093,
			-0.00085479386629692035,
			0.00012046344737264886,
			-1.4795855767566064e-5,
			1.5310764499971662e-6,
			-1.0478858511964396e-7,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate max error: 3.23342e-11
		inline static std::array<double, MAX_TERMS> s_ErfApproximateCoefficients = {
			1.1283791670631784,
			-0.37612638483953933,
			0.11283782748887524,
			-0.026865445893359986,
			0.0052210750172173271,
			-0.00084847126956686805,
			0.00011270548762792266,
			-9.6801370520145687e-6,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};


		// ERFC
		// For small x: erfc(x) = 1 - erf(x), reuses s_Erf*Coefficients.
		// For large x (|x| >= 4): asymptotic expansion.
		//   erfc(x) ~ exp(-x^2)/sqrt(pi) * Sig (-1)^n * (2n-1)!! / (2x)^(2n+1) * (1/x)
		// Coefficients below are the asymptotic series in powers of (1/x^2),
		// multiplied out as a polynomial in u = 1/x^2.
		// Evaluate as: (exp(-x^2) / (x*sqrt pi)) * Horner(1/x^2, c)

		inline static std::array<double, MAX_TERMS> s_ErfcAsymptoticExactCoefficients = {
			 1.0,                       //  (-1)^0 * (2*0-1)!! / 2^0 = 1
			-0.5,                       //  (-1)^1 * 1!! / 2^1
			 0.75,                      //  (-1)^2 * 3!! / 2^2
			-1.875,                     //  (-1)^3 * 5!! / 2^3
			 6.5625,                    //  (-1)^4 * 7!! / 2^4
			-29.53125,                  //  (-1)^5 * 9!! / 2^5
			 162.421875,                //  (-1)^6 * 11!! / 2^6
			-1055.7421875,              //  (-1)^7 * 13!! / 2^7
			 7918.06640625,             //  (-1)^8 * 15!! / 2^8
			-67303.564453125,           //  (-1)^9 * 17!! / 2^9
			 0.0,                       //  diverges rapidly beyond here; truncate
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// Balanced: first 6 terms of asymptotic expansion
		inline static std::array<double, MAX_TERMS> s_ErfcAsymptoticBalancedCoefficients = {
			1.0000000000000000,
			-0.50000000000000000,
			0.75000000000000000,
			-1.8750000000000000,
			6.5625000000000000,
			-29.531250000000000,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// Approximate: first 4 terms of asymptotic expansion
		inline static std::array<double, MAX_TERMS> s_ErfcAsymptoticApproximateCoefficients = {
			1.0000000000000000,
			-0.50000000000000000,
			0.75000000000000000,
			-1.8750000000000000,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0
		};

		// LGAMMA / TGAMMA
		// Not a Taylor series. Uses Lanczos approximation with g = 7.
		// Gamma(z+1) = sqrt (2pi) * (z + g + 0.5)^(z+0.5) * exp(-(z+g+0.5)) * A_g(z)
		// A_g(z) = c[0] + Sig c[n] / (z + n)   for n = 1..N
		// 9-term rational sum (9 Lanczos coefficients). Remaining entries are 0.0.
		// lgamma is log(|Gamma(x)|); tgamma is Gamma(x) directly - both use same coefficients.

		inline static std::array<double, MAX_TERMS> s_LanczosExactCoefficients = {
			 0.99999999999980993227,    // c[0]
		   676.52036812188509856,       // c[1]
		 -1259.13921672240287085,       // c[2]
		   771.32342877765307879,       // c[3]
		  -176.61502916214059157,       // c[4]
			12.50734327868690481,       // c[5]
			-0.13857109526572011689,    // c[6]
			 0.0000098650755802898471,  // c[7]
			 0.0000001505143016942768,  // c[8]
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0,
			 0.0
		};

		// lgamma and tgamma share the same Lanczos coefficients.
		// lgamma adds log and sign tracking on top; tgamma exponentiates directly.
		inline static std::array<double, MAX_TERMS> s_LanczosBalancedCoefficients = {};
		inline static std::array<double, MAX_TERMS> s_LanczosApproximateCoefficients = {};
	};
}