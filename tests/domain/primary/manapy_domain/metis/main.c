/*
 * Build with:  gcc -O2 metis_driver.c -o metis_driver -lmetis
 *
 * Usage:
 *   Grid        : ./metis_driver grid   <width> <height> <nparts>
 *   Random graph: ./metis_driver random <nvtxs> <degree> <nparts> [seed]
 *
 * Prints the edge-cut returned by METIS.  For small graphs (≤50 vertices)
 * it also prints the partition vector so you can inspect the result.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <metis.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* 1.  GRID  (width × height, undirected, 4-neighbour lattice)        */
static void generate_grid(idx_t w, idx_t h,
                          idx_t **xadj, idx_t **adjncy, idx_t *nnz)
{
    idx_t nvtxs = w * h;                  /* total vertices                */
    size_t cap  = (size_t)nvtxs * 4;      /* worst-case adjacency entries   */

    *xadj   = malloc((nvtxs + 1) * sizeof(idx_t));
    *adjncy = malloc(cap * sizeof(idx_t));
    if (!*xadj || !*adjncy) { perror("malloc"); exit(EXIT_FAILURE); }

    idx_t k = 0;               /* running index into adjncy           */
    (*xadj)[0] = 0;

    for (idx_t y = 0; y < h; ++y) {
        for (idx_t x = 0; x < w; ++x) {
            idx_t v = y * w + x;

            if (y > 0)        (*adjncy)[k++] = (y - 1) * w + x;      /* up   */
            if (x > 0)        (*adjncy)[k++] =  y      * w + x - 1;  /* left */
            if (x < w - 1)    (*adjncy)[k++] =  y      * w + x + 1;  /* right*/
            if (y < h - 1)    (*adjncy)[k++] = (y + 1) * w + x;      /* down */

            (*xadj)[v + 1] = k;
        }
    }
    *nnz = k;
    *adjncy = realloc(*adjncy, k * sizeof(idx_t));   /* trim to fit */
}

/* ------------------------------------------------------------------ */
/* 2.  RANDOM fixed-degree (undirected)                               */
static void generate_random(idx_t nvtxs, int degree,
                            idx_t **xadj, idx_t **adjncy, idx_t *nnz)
{
    if (degree >= nvtxs) { fprintf(stderr, "degree too high\n"); exit(EXIT_FAILURE); }

    size_t cap = (size_t)nvtxs * degree * 2;      /* each edge stored twice */
    *xadj   = malloc((nvtxs + 1) * sizeof(idx_t));
    *adjncy = malloc(cap * sizeof(idx_t));
    if (!*xadj || !*adjncy) { perror("malloc"); exit(EXIT_FAILURE); }

    idx_t k = 0;
    (*xadj)[0] = 0;

    /* naive O(n·degree²) duplicate checking – fine for testing */
    for (idx_t v = 0; v < nvtxs; ++v) {
        int added = 0;
        while (added < degree) {
            idx_t u = rand() % nvtxs;
            if (u == v) continue;                     /* no self-loops */

            /* duplicate? */
            int dup = 0;
            for (idx_t i = (*xadj)[v]; i < k; ++i)
                if ((*adjncy)[i] == u) { dup = 1; break; }
            if (dup) continue;

            /* add both directions */
            (*adjncy)[k++] = u;
            (*xadj)[v + 1] = k;       /* provisional; will shift later */

            (*adjncy)[k++] = v;       /* u -> v, store for symmetry   */
            added++;
        }
    }

    /* fix xadj offsets (second pass) */
    idx_t shift = 0;
    for (idx_t v = 0; v < nvtxs; ++v) {
        idx_t start = (*xadj)[v] + shift;
        idx_t end   = start + degree;
        (*xadj)[v]   = start;
        shift       += degree;                           /* invariant */
        (*xadj)[v + 1] = end;
    }
    *nnz = k;
    *adjncy = realloc(*adjncy, k * sizeof(idx_t));
}

/* ------------------------------------------------------------------ */
static void partition_with_metis(idx_t nvtxs, idx_t *xadj, idx_t *adjncy,
                                 idx_t nparts)
{
    idx_t ncon  = 1;
    idx_t objval;
    idx_t *part = malloc(nvtxs * sizeof(idx_t));
    if (!part) { perror("malloc"); exit(EXIT_FAILURE); }

    int status = METIS_PartGraphKway(&nvtxs, &ncon,
                                     xadj, adjncy,
                                     NULL, NULL, NULL,     /* no weights */
                                     &nparts,
                                     NULL, NULL,           /* defaults   */
                                     NULL,                 /* options    */
                                     &objval, part);

    if (status != METIS_OK) {
        fprintf(stderr, "METIS failed (status=%d)\n", status);
        exit(EXIT_FAILURE);
    }

    printf("Edge-cut: %d\n", (int)objval);

    if (nvtxs <= 50) {           /* don’t spam for huge graphs */
        puts("Partition vector:");
        for (idx_t i = 0; i < nvtxs; ++i)
            printf("  v%4d → %d\n", (int)i, (int)part[i]);
    }

    free(part);
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Grid   : %s grid   <width> <height> <nparts>\n"
            "Random : %s random <nvtxs> <degree> <nparts> [seed]\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    /* optional RNG seed */
    if (argc == 6) srand((unsigned)strtoul(argv[5], NULL, 10));
    else            srand((unsigned)time(NULL));

    idx_t *xadj = NULL, *adjncy = NULL, nnz = 0;
    idx_t nparts;

    if (strcmp(argv[1], "grid") == 0) {
        idx_t w      = strtoll(argv[2], NULL, 10);
        idx_t h      = strtoll(argv[3], NULL, 10);
        nparts       = strtoll(argv[4], NULL, 10);
        idx_t nvtxs  = w * h;

        generate_grid(w, h, &xadj, &adjncy, &nnz);
        printf("Generated %lld×%lld grid (%lld vertices, %lld edges)\n",
               (long long)w, (long long)h,
               (long long)nvtxs, (long long)(nnz / 2));

        partition_with_metis(nvtxs, xadj, adjncy, nparts);

    } else if (strcmp(argv[1], "random") == 0) {
        idx_t nvtxs  = strtoll(argv[2], NULL, 10);
        int   degree = atoi(argv[3]);
        nparts       = strtoll(argv[4], NULL, 10);

        generate_random(nvtxs, degree, &xadj, &adjncy, &nnz);
        printf("Generated random graph (%lld vertices, %lld edges, degree=%d)\n",
               (long long)nvtxs, (long long)(nnz / 2), degree);

        partition_with_metis(nvtxs, xadj, adjncy, nparts);

    } else {
        fprintf(stderr, "first argument must be 'grid' or 'random'\n");
        return EXIT_FAILURE;
    }

    free(xadj);
    free(adjncy);
    return EXIT_SUCCESS;
}
