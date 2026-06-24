import cooler
import numpy as np

# Load cooler file
# c = cooler.Cooler("t_cells/tads/GSM6705657_CD4_Q4_rep1_250kb.cool")

# # Compute balancing weights if missing
# # if "weight" not in c.bins().columns:
# # 	cooler.balance_cooler(c, store=True)

# # Fetch ICE-balanced (normalized) matrix for chr2
# mat = c.matrix(balance=False).fetch("chr2")

# # Replace NaNs (unmappable bins) with 0
# mat = np.nan_to_num(mat, nan=0.0)

# # Save to file
# np.savetxt("t_cells/tads/CD4_Q4_chr2_matrix_250kb.txt", mat)



# Get bins table
# bins = c.bins()[:]

# # Get chromosome sizes
# chrom_sizes = bins.groupby("chrom")["end"].max()

# # Save to file
# chrom_sizes.to_csv("genome.chrom.sizes", sep="\t", header=False)

# print(chrom_sizes)


# import cooler
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import SymLogNorm

# # Load Hi-C data (10kb resolution)
# c = cooler.Cooler("t_cells/tads/GSM6705663_DP_rep1.cool")

# # Resolutions
# cool_res = 10_000
# tad_res = 250_000
# scale_factor = tad_res // cool_res   # 25

# chrom = 'chr2'

# # -------- Define window in 250kb bins --------
# start_bin_250, end_bin_250 = 400, 600

# # Convert window to 10kb bins
# start_bin_10 = start_bin_250 * scale_factor
# end_bin_10   = end_bin_250 * scale_factor

# # Fetch chromosome matrix and slice in 10kb space
# full_matrix = c.matrix(balance=False).fetch(chrom)
# full_matrix = np.nan_to_num(full_matrix, nan=0.0)

# matrix = full_matrix[start_bin_10:end_bin_10,
#                      start_bin_10:end_bin_10]

# # -------- Overlay TADs (convert 250kb → 10kb) --------
# tad_file = 't_cells/tads/inferred_internal_tads/node_0_250kb.tad'
# tads = []

# with open(tad_file) as f:
#     for line in f:
#         s250, e250 = list(map(float, line.strip().split()))[:2]

#         if e250 < start_bin_250 or s250 > end_bin_250:
#             continue

#         # Clip in 250kb space
#         s_clip_250 = max(s250, start_bin_250)
#         e_clip_250 = min(e250, end_bin_250)

#         # Convert to 10kb bins
#         s10 = int(s_clip_250 * scale_factor)
#         e10 = int(e_clip_250 * scale_factor)

#         # Convert to pixel coordinates
#         s_pix = s10 - start_bin_10
#         e_pix = e10 - start_bin_10

#         tads.append((s_pix, e_pix))

# # -------- Plot (same scaling as before) --------
# plt.figure(figsize=(10,10))

# vmax = np.percentile(matrix[matrix > 0], 95)


# plt.imshow(matrix,
#            origin='lower',
#            cmap='Reds',
#            norm=SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=vmax))


# # White TAD borders
# for s_pix, e_pix in tads:
#     plt.plot([s_pix, e_pix, e_pix, s_pix, s_pix],
#              [s_pix, s_pix, e_pix, e_pix, s_pix],
#              color='black', linewidth=1.2)

# plt.xlabel("10kb bins")
# plt.ylabel("10kb bins")
# plt.title(f"{chrom}: 250kb bins {start_bin_250}-{end_bin_250}")
# plt.colorbar(label='Contacts (log-sym)')

# plt.savefig("t_cells/tads/chr2_node_0_250kb_fine.png",
#             dpi=150, bbox_inches='tight')

# plt.close()





import hicstraw

hic = hicstraw.HiCFile("hemato_cells/GSE152918_GMP.hic")

print("Chromosomes:")
for c in hic.getChromosomes():
    print(c.name, c.length)

print("Resolutions:")
print(hic.getResolutions())


