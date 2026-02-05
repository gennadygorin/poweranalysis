import numpy as np, scipy.stats, matplotlib.pyplot as plt, streamlit as st

st.set_page_config(
    page_title="DEPower",
    page_icon="logo.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Work Sans', sans-serif;
}

header[data-testid="stHeader"] {
    display: none;
}

/* Force green primary color in all themes */
div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] + div {
    border-color: #6BBD84 !important;
}
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    border-color: #6BBD84 !important;
}
div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
    background-color: #6BBD84 !important;
    border-color: #6BBD84 !important;
}
div[data-baseweb="input"] {
    border-color: #6BBD84 !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #6BBD84 !important;
    box-shadow: #6BBD84 0px 0px 0px 1px !important;
}

.stButton button[kind="primary"] {
    background-color: #6BBD84 !important;
    border-color: #6BBD84 !important;
}
.stButton button[kind="primary"]:hover {
    background-color: #5AAC73 !important;
    border-color: #5AAC73 !important;
}

.content-section {
    background-color: #69ad83;
    color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 0.75rem;
    margin: 2rem 0;
}
.content-section h2 {
    color: #FFFFFF;
    margin-top: 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>DEPower: DESeq2 Approximate Sample Size Calculator</h1>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Mode", ["Bulk","Single-cell"],horizontal=True)
    if mode=="Single-cell":
        n_cells = st.number_input("Number of cells", value=7500, min_value=1, step=100,format='%u')
        cell_pct = st.number_input("Cell type %", value=10.0, min_value=0.0001, step=1.0,format='%g')
        avg_expr = st.number_input("Average expression in cell type (UMIs per barcode)", value=0.1, min_value=1e-6, step=0.01,format='%g')
        base_mean = n_cells*(cell_pct/100.0)*avg_expr
    else:
        base_mean = st.number_input("Base expression (raw counts, bulk)", value=100.0, min_value=1e-6, step=1.0,format='%g')

    alpha_raw = st.number_input("Statistical significance threshold (alpha)", value=0.05, min_value=1e-300, step=0.01,format='%g')
    l2fc = st.number_input("log2 fold change", value=1.0, step=0.1,format='%g')
    fdir_mode = st.radio("False discovery correction", ["Benjamini-Hochberg","Bonferroni"],horizontal=True)
    if fdir_mode=="Bonferroni":
        n_genes = st.number_input("Number of comparisons (genes in analysis)", value=20000, min_value=1, step=100,format='%u')
    elif fdir_mode=="Benjamini-Hochberg":
        n_genes_raw = st.number_input("Number of comparisons (genes in analysis)", value=20000, min_value=1, step=100,format='%u')
        q = st.number_input("Expected % genes differentially expressed", value=10.0, min_value=0.0, max_value=100.0, step=2.0,format='%g')
        q = np.clip(q/100,1/n_genes_raw,1)
        n_genes = n_genes_raw*q

    if "computed" not in st.session_state:
        st.session_state.computed = False

    if st.button("Compute",type="primary"):
        st.session_state.computed = True

def compute_n(base_condition_mean, alpha_raw, l2fc, n_genes):
    alpha = alpha_raw/n_genes
    lfc = l2fc/np.log2(np.e)
    wald = scipy.stats.norm.isf(alpha/2.0)
    se = lfc/wald
    se2 = se**2
    m0 = base_condition_mean
    m1 = m0*np.exp(lfc)
    baseMean = (m1+m0)/2.0
    dispersion = 10**(-1.5) + 10**(0.5)/baseMean
    disp_delta = 10**(0.5)
    def disp_corr(m,d): return m/(1+m*d)
    M0t = disp_corr(m0, dispersion); M1t = disp_corr(m1, dispersion)
    n = int(np.ceil((1/M0t + 1/M1t)/se2))
    M0t, M1t = disp_corr(m0, 1e-6), disp_corr(m1, 1e-6)
    n_theo_min = int(np.ceil((1/M0t + 1/M1t)/se2))
    M0t, M1t = disp_corr(m0, dispersion/disp_delta), disp_corr(m1, dispersion/disp_delta)
    n_theo_opt = int(np.ceil((1/M0t + 1/M1t)/se2))
    M0t, M1t = disp_corr(m0, dispersion*disp_delta), disp_corr(m1, dispersion*disp_delta)
    n_theo_pes = int(np.ceil((1/M0t + 1/M1t)/se2))
    return n, n_theo_min, n_theo_opt, n_theo_pes, dispersion

col2_placeholder = col2.empty()

if st.session_state.computed:
    try:
        n, n_min, n_opt, n_pes, dispersion = compute_n(base_mean, alpha_raw, l2fc, n_genes)

        disp_delta = 10**(0.5)
        dispersions = np.logspace(-2, 2, 400)
        lfc = l2fc/np.log2(np.e)
        wald = scipy.stats.norm.isf((alpha_raw/n_genes)/2.0)
        se2 = (lfc/wald)**2
        m0 = base_mean; m1 = m0*np.exp(lfc)
        def disp_corr(m,d): return m/(1+m*d)
        M0t = disp_corr(m0, dispersions); M1t = disp_corr(m1, dispersions)
        ngrid = np.ceil((1/M0t + 1/M1t)/se2).astype(int)

        fig1,ax1=plt.subplots(1,2,figsize=(8,5))
        nmax = int(n_pes*2)
        ngrid = np.linspace(1,nmax,100).astype(int) if nmax>100 else np.arange(1,nmax)
        def get_logp(m0,m1,dispersion,n,n_genes):
            M0t,M1t = disp_corr(m0,dispersion),disp_corr(m1,dispersion)
            se = np.sqrt((1/M0t+1/M1t)/n)
            wald = lfc/se
            log10p = np.log10(2)+scipy.stats.norm.logsf(np.abs(wald))/np.log(10)
            log10p = np.minimum(np.log10(n_genes)+log10p,0)
            return log10p

        def get_p(m0,m1,dispersion,n,n_genes):
            M0t,M1t = disp_corr(m0,dispersion),disp_corr(m1,dispersion)
            se = np.sqrt((1/M0t+1/M1t)/n)
            wald = lfc/se
            p = 2*scipy.stats.norm.sf(np.abs(wald))
            p = np.minimum(n_genes*p,1)
            return p

        pgrid = get_logp(m0,m1,dispersion,ngrid,n_genes)
        ax1[0].scatter(ngrid,-pgrid,3,'k')
        pgrid_l = get_logp(m0,m1,dispersion/disp_delta,ngrid,n_genes)
        pgrid_h = get_logp(m0,m1,dispersion*disp_delta,ngrid,n_genes)
        ax1[0].fill_between(ngrid,-pgrid_l,-pgrid_h,alpha=0.2,step='post',edgecolor=None)
        ax1[0].scatter(n,-get_logp(m0,m1,dispersion,n,n_genes),50,'indianred',zorder=10000)
        ax1[0].set_ylabel(r'Adjusted $p$-value')

        ax1[0].set_yscale('log')
        ax1[0].set_xlabel('Number of replicates')
        ax1[0].plot([1,nmax],[-np.log10(alpha_raw)]*2,'m--')

        yl=ax1[0].get_ylim()
        ticks = np.array(ax1[0].get_yticks())
        labels = -ticks.copy()
        labels = [str((10**l).round(2)) if l>-3 \
            else r'$10^{'+str(int(np.floor(l)) if l != 0 else 0)+'}$' for l in labels ]
        ax1[0].set_yticks(ticks,labels=labels)
        ax1[0].set_ylim(yl)

        dispersions=np.logspace(-2,1.2,1000)
        M0t,M1t = disp_corr(m0,dispersions),disp_corr(m1,dispersions)
        ngrid = np.ceil((1/M0t+1/M1t)/se2).astype(int)
        ax1[1].plot(dispersions,ngrid,'k',linewidth=2)

        ax1[1].scatter(dispersion,n,50,'indianred',zorder=10000)
        ax1[1].fill_between([dispersion*disp_delta,dispersion/disp_delta],[n_min]*2,[ngrid.max()]*2,alpha=0.2)
        ax1[1].plot(dispersions,np.ones_like(dispersions)*n_min,'r--')
        ax1[1].set_xscale('log')
        ax1[1].set_yscale('log')
        ax1[1].set_xlabel('Dispersion')
        ax1[1].set_ylabel('Number of replicates')
        fig1.tight_layout()

        with col2_placeholder.container():
            st.write(f"Expected # samples (approximate dispersion): {n}")
            st.write(f"Theoretical minimum (zero dispersion): {n_min}")
            st.write(f"Low dispersion (optimistic): {n_opt}")
            st.write(f"High dispersion (pessimistic): {n_pes}")
            st.pyplot(fig1)
    except Exception as e:
        with col2_placeholder.container():
            st.error(f"Input error: {e}")
else:
    with col2_placeholder.container():
        st.markdown("""
            <style>
            .logo-spacer {
                height: 20px;
            }
            @media (min-width: 768px) {
                .logo-spacer {
                    height: 150px;
                }
            }
            </style>
            <div class="logo-spacer"></div>
        """, unsafe_allow_html=True)
        col2_spacer1, col2_center, col2_spacer2 = st.columns([1, 3, 1])
        with col2_center:
            st.image("logo.svg", width=500)

st.markdown(
    """
<div class="content-section">

## The big picture
This calculator provides a simple interface to calculate
the sample size for a RNA sequencing experiment.

To calculate the sample size, we need to specify some
information about the *smallest* differential expression
we would like to be able to detect using *DESeq2*:

* The **base expression**: how many reads do we expect to see
in our base condition? Higher-expressed genes are less
common, but easier to analyze.
* The **statistical significance threshold**:
what $p$-value
cutoff will be used for statistical testing? Lower $p$-values
increase confidence but require more samples.
* The **effect size**: what $\log_2$ fold change would we
like to detect? Stronger changes (positive or negative) are
easier to detect.

For pseudobulk single-cell analysis, the base expression comes
from

* The **number of cells**: how many cells will the experiment
capture? This is determined by the technology and
experiment efficiency.
* The **cell type %**: how common is the cell type of
interest? Rarer cell types require more samples.
* The **average expression in the cell type of interest**:
how highly expressed is the gene? Higher-expressed genes are
much less common, but easier to analyze.

Experiment design should account for multiple
testing. By default, *DESeq2* uses the Benjamini-Hochberg
correction. To predict the sample size, we need to make an
educated guess about

* The **number of comparisons**: how many genes will we be
comparing? Typically, this is a large fraction of the genome.
* The **% of genes that show differential expression**: how
many genes do we expect to show differential expression
*stronger* than the minimum we are looking for?

We may also use the more conservative Bonferroni method, which
only requires the **number of comparisons**.

</div>

## Example

Consider designing an RNA-seq experiment powered to detect a
two-fold increase ($\log_2$ fold change = 1) with a
$p$-value of 0.05.

Suppose we primarily care about relatively
high-expression genes with about 100 reads per sample, and
expect about 10% of genes to be differentially expressed.

For a genome-wide screen, we would expect to compare about
20,000 genes. Inputting these numbers, we find such an
experiment would typically require **about 5 samples**.

However, our ability to detect changes in genes is governed by
the spread, or dispersion, between samples in a single
condition.
This dispersion is loosely governed by the expression level,
but can vary quite a bit. If we take the optimistic lower
estimate, we need **about 2 samples**; if we take a pessimistic
higher estimate, we need **about 13 samples**.

We may also want to detect *declines* in gene expression.
If we check the power for a $\log_2$ fold change of -1,
a twofold decrease, we find we need **about 7 samples**, with
**3 to 19 samples** in the optimistic and pessimistic
scenarios. These numbers are higher because we learn less
about the gene expression from lower numbers.

Therefore, to detect such expression differences with a
reasonable margin of error for variability between samples and
multiple testing, we would want to collect **about 20 samples
per condition**, or **about 40 samples total**.

## The statistical foundations

The calculator implements a simplified version of the *DESeq2*
two-level contrast. The following assumptions are made:

* **Negligible depth differences between levels**: this is
generally typical.
* **Simplified expression-dispersion relation**: the dispersion
is estimated from an order-of-magnitude curve typical of
RNA-seq experiments. The upper and lower estimates are obtained
by considering the curve $\pm$ half an order of magnitude.
* **No batch effects**: batch variability can be considerable,
cannot be constrained or estimated, and should be estimated
using pilot experiments.
* **No outliers**: the calculator assumes the ideal-case
scenario where all samples are comparable and successfully
sequenced. In practice, a greater margin of error may be
necessary.
* **Identical cell type fractions**: the
calculator does not account for the sometimes substantial
differences in cell type conditions observed across replicates
or conditions. For a margin of error, it may be appropriate
to use the smallest cell type fraction expected to be observed
in *any* of the samples.

The baseline expression may be more naturally expressed in
transcripts per million (TPM), a compositional measure
(as opposed to raw read counts). For instance, if we are
interested in detecting DE for genes that are expressed at 10 TPM
in the tissue under consideration, and plan to run an experiment
at a depth of 30 million reads, the resulting read count is 300.
However, this heavily depends on the alignment efficiency, which
is typically around 50%, so 150 reads is a safer assumption.
""",
    unsafe_allow_html=True,
)