import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Sarcasm Detection Using Traditional Machine Learning" },
      { name: "description", content: "Comparing Logistic Regression and Linear SVM on 26,602 news headlines — and uncovering perfect publisher-label bias." },
      { property: "og:title", content: "Sarcasm Detection Using Traditional ML" },
      { property: "og:description", content: "TF-IDF + Linear SVM reaches 85.19% accuracy on news headlines — but the dataset hides a catch." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const test = [
  ["Accuracy", "84.78%", "85.19%"],
  ["Precision", "82.82%", "82.70%"],
  ["Recall", "82.32%", "83.69%"],
  ["F1-score", "82.57%", "83.19%"],
];
const cv = [
  ["Accuracy", "84.40% ± 0.59", "84.74% ± 0.54"],
  ["Precision", "83.00% ± 0.65", "82.95% ± 0.91"],
  ["Recall", "80.98% ± 0.76", "82.03% ± 0.32"],
  ["F1-score", "81.98% ± 0.70", "82.48% ± 0.55"],
];
const stats = [
  ["26,602", "unique headlines"],
  ["14,951", "non-sarcastic"],
  ["11,651", "sarcastic"],
  ["80 / 20", "stratified split"],
];
const steps = [
  ["Data preparation", "Selected headline and label, removed duplicates and conflicting records, checked for missing values."],
  ["TF-IDF features", "Unigrams + bigrams, min_df = 2, up to 50,000 features, sublinear TF scaling."],
  ["Model development", "Scikit-learn pipelines: Logistic Regression baseline vs Linear Support Vector Machine."],
  ["Evaluation", "Accuracy, precision, recall, F1, confusion matrices and five-fold stratified cross-validation."],
];
const limits = [
  "Publisher identity is perfectly associated with the target labels.",
  "The model may learn source-related vocabulary and writing patterns.",
  "Sarcasm depends on context, tone and cultural knowledge TF-IDF cannot fully capture.",
  "Random splits don't establish generalisation to unseen publishers.",
  "Results are not proof of general-purpose sarcasm understanding.",
];
const future = [
  "Datasets with both classes across multiple publishers",
  "Source-aware and publisher-held-out evaluation",
  "Comparison with contextual language models",
  "Explainability and influential linguistic features",
  "NLP classification for cybersecurity text analytics",
];
const stack = ["Python", "Google Colab", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "HF Datasets", "Joblib", "Streamlit"];

function Section({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="rule-double py-12">
      <div className="grid gap-8 md:grid-cols-[200px_1fr]">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-primary">§ {n}</p>
          <h2 className="mt-2 font-display text-3xl font-bold leading-tight">{title}</h2>
        </div>
        <div>{children}</div>
      </div>
    </section>
  );
}

function Table({ rows, caption }: { rows: string[][]; caption: string }) {
  return (
    <div>
      <p className="mb-3 font-mono text-xs uppercase tracking-widest text-muted-foreground">{caption}</p>
      <table className="w-full border-collapse font-mono text-sm">
        <thead>
          <tr className="border-b-2 border-foreground text-left">
            <th className="py-2 font-semibold">Metric</th>
            <th className="py-2 text-right font-semibold">Log. Regression</th>
            <th className="py-2 text-right font-semibold text-primary">Linear SVM</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([m, a, b]) => (
            <tr key={m} className="border-b border-border">
              <td className="py-2">{m}</td>
              <td className="py-2 text-right">{a}</td>
              <td className="py-2 text-right font-semibold text-primary">{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Index() {
  return (
    <main className="mx-auto max-w-6xl px-6 pb-20">
      <header className="flex items-center justify-between border-b border-foreground py-3 font-mono text-xs uppercase tracking-widest">
        <span>NLP Research Report</span>
        <span className="hidden sm:inline">Vol. I · News Headlines</span>
        <span className="text-primary">Status: Complete</span>
      </header>

      <div className="fade-up py-16 text-center">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-muted-foreground">Breaking · Supervised Classification</p>
        <h1 className="mx-auto mt-6 max-w-4xl font-display text-5xl font-black leading-[1.05] md:text-7xl">
          Sarcasm Detection Using <em className="text-primary">Traditional</em> Machine Learning
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
          Can TF-IDF and a linear classifier tell <span className="italic">The Onion</span> from the news? Mostly — but the
          dataset itself had a secret.
        </p>
        <div className="mx-auto mt-10 inline-flex flex-wrap justify-center gap-px border border-foreground bg-foreground">
          {[["0", "Non-sarcastic"], ["1", "Sarcastic"]].map(([k, v]) => (
            <div key={k} className="flex items-center gap-3 bg-card px-6 py-3">
              <span className="font-mono text-2xl font-semibold text-primary">{k}</span>
              <span className="font-mono text-xs uppercase tracking-widest">{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 border-y-2 border-foreground md:grid-cols-4">
        {stats.map(([v, l], i) => (
          <div key={l} className={`p-6 ${i > 0 ? "md:border-l" : ""} ${i % 2 ? "border-l" : ""} border-border`}>
            <p className="font-display text-4xl font-black">{v}</p>
            <p className="mt-1 font-mono text-xs uppercase tracking-widest text-muted-foreground">{l}</p>
          </div>
        ))}
      </div>

      <div className="mt-12" />
      <Section n="01" title="Dataset">
        <p className="text-lg leading-relaxed">
          The <strong>News Headlines Dataset for Sarcasm Detection</strong> by Rishabh Misra — 26,709 headlines with
          <code className="mx-1 font-mono text-sm">article_link</code>,<code className="mx-1 font-mono text-sm">headline</code> and
          <code className="mx-1 font-mono text-sm">is_sarcastic</code>. After removing duplicates and conflicting labels, 26,602 unique
          headlines remained: 21,281 for training and 5,321 for testing.
        </p>
      </Section>

      <Section n="02" title="Methodology">
        <ol className="grid gap-px border border-border bg-border sm:grid-cols-2">
          {steps.map(([t, d], i) => (
            <li key={t} className="bg-card p-6">
              <span className="font-mono text-sm text-primary">0{i + 1}</span>
              <h3 className="mt-1 font-display text-xl font-bold">{t}</h3>
              <p className="mt-2 text-muted-foreground">{d}</p>
            </li>
          ))}
        </ol>
      </Section>

      <Section n="03" title="Results">
        <div className="grid gap-10 lg:grid-cols-2">
          <Table rows={test} caption="Held-out test set" />
          <Table rows={cv} caption="Five-fold cross-validation" />
        </div>
        <p className="mt-8 border-l-4 border-primary pl-4 text-lg">
          <strong>Linear SVM</strong> achieved the strongest accuracy, recall and F1-score and was selected as the preferred model.
        </p>
      </Section>

      <Section n="04" title="The Catch: Publisher Bias">
        <div className="grid gap-6 md:grid-cols-2">
          {[["Huffington Post", "14,985", "0"], ["The Onion", "0", "11,724"]].map(([p, n, s]) => (
            <div key={p} className="border-2 border-foreground bg-card p-6">
              <h3 className="font-display text-2xl font-bold italic">{p}</h3>
              <div className="mt-4 flex justify-between font-mono text-sm">
                <span>Non-sarcastic <b className="block text-2xl">{n}</b></span>
                <span className="text-right text-primary">Sarcastic <b className="block text-2xl">{s}</b></span>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 bg-accent p-6">
          <p className="font-display text-2xl font-bold">
            A publisher-only classifier hit <span className="text-primary">100%</span> accuracy — versus a 56.10% majority baseline.
          </p>
          <p className="mt-3">
            Publisher identity alone predicts the label without reading the headline. The headline model's score therefore can't prove it
            learned sarcasm independently of source-specific language.
          </p>
        </div>
      </Section>

      <Section n="05" title="Limitations">
        <ul className="space-y-3">
          {limits.map((l, i) => (
            <li key={l} className="flex gap-4 border-b border-border pb-3">
              <span className="font-mono text-primary">{i + 1}.</span>
              <span>{l}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section n="06" title="Future Work">
        <ul className="grid gap-3 sm:grid-cols-2">
          {future.map((f) => (
            <li key={f} className="border border-foreground p-4">→ {f}</li>
          ))}
        </ul>
      </Section>

      <Section n="07" title="Tools">
        <div className="flex flex-wrap gap-2">
          {stack.map((s) => (
            <span key={s} className="border border-foreground px-3 py-1 font-mono text-sm">{s}</span>
          ))}
        </div>
        <p className="mt-6 text-muted-foreground">
          A Streamlit app is planned to demo headline classification using the saved Linear SVM pipeline.
        </p>
      </Section>

      <footer className="rule-double pt-6 text-center font-mono text-xs uppercase tracking-widest text-muted-foreground">
        Supervised NLP · Model evaluation · Dataset bias · Generalisation
      </footer>
    </main>
  );
}
