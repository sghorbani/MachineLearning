

using System.Text.RegularExpressions;


string path = "housing.csv";

var dataLines = File
	.ReadAllLines(path)
	.Where(line => !string.IsNullOrWhiteSpace(line))
	.Select(line => Regex
					  .Split(line.Trim(), @"\s+")
					  .Select(v => double.Parse(v))
					  .ToArray()
	)
	.ToList();

int m = dataLines.Count;
int totalColums = dataLines[0].Length;

int medvIndex = totalColums - 1;
int lstatIndex = totalColums - 2;

double[] y = [.. dataLines.Select(row => row[medvIndex])];


Console.WriteLine("****** just with lstat column: ");
var X_lstat = dataLines
	.Select(row => new double[] { row[lstatIndex] })
	.ToArray();
RunGradientDescent(X_lstat, y);

Console.WriteLine("****** with hole features:");
var X_all = dataLines
	.Select(r => r.Take(totalColums - 1).ToArray())
	.ToArray();
RunGradientDescent(X_all, y);
Console.ReadLine();

static (double[][] X_norm, double[] means, double[] stds) FeatureNormalize(double[][] X)
{
	int m = X.Length;
	int n = X[0].Length;
	var means = new double[n];
	var stds = new double[n];
	var X_norm = new double[m][];

	for (int j = 0; j < n; j++)
	{
		means[j] = X.Select(row => row[j]).Average();
		stds[j] = Math.Sqrt(X.Select(r => Math.Pow(r[j] - means[j], 2)).Sum() / m);
	}

	for (int i = 0; i < m; i++)
	{
		X_norm[i] = new double[n];
		for (int j = 0; j < n; j++)
		{
			if (stds[j] != 0)
				X_norm[i][j] = (X[i][j] - means[j]) / stds[j];
			else
				X_norm[i][j] = 0;
		}
	}

	return (X_norm, means, stds);
}

static void RunGradientDescent(double[][] X_raw, double[] y)
{
	int m = y.Length;
	int n = X_raw[0].Length;

	double[][] X_normalized;

	var (X_norm, means, stds) = FeatureNormalize(X_raw);
	X_normalized = X_norm;


	var X = new double[m][];
	for (int i = 0; i < m; i++)
	{
		X[i] = new double[n + 1];
		X[i][0] = 1.0;
		for (int j = 0; j < n; j++)
			X[i][j + 1] = X_normalized[i][j];
	}


	var w = new double[n + 1];
	double alpha = 0.001;
	int iters = 1500;

	GradientDescent(X, y, w, alpha, iters);


	double sse = 0;
	for (int i = 0; i < m; i++)
	{
		double pred = 0;
		for (int j = 0; j < n + 1; j++)
			pred += X[i][j] * w[j];
		sse += Math.Pow(pred - y[i], 2);
	} 
	Console.WriteLine($"samples count: {m}  | features count : {n}");
	Console.WriteLine($"rating late : {alpha}  | iters: {iters}");
	Console.WriteLine($" w total: {Environment.NewLine} [{string.Join(", ", w.Select(t => t.ToString("F4")))}]");
	Console.WriteLine($"SSE = {sse:F4}");
	Console.WriteLine(Environment.NewLine); 
}

static void GradientDescent(double[][] X, double[] y, double[] w, double alpha, int iterations)
{
	int m = y.Length;
	int n = w.Length;

	for (int iter = 0; iter < iterations; iter++)
	{
		var grad = new double[n];
		var h = new double[m];

		for (int i = 0; i < m; i++)
		{
			double sum = 0;
			for (int j = 0; j < n; j++)
				sum += X[i][j] * w[j];
			h[i] = sum;
		}

		for (int j = 0; j < n; j++)
		{
			double s = 0;
			for (int i = 0; i < m; i++)
				s += (h[i] - y[i]) * X[i][j];
			grad[j] = s / m;
		}


		for (int j = 0; j < n; j++)
			w[j] -= alpha * grad[j];
	}
}