#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    // 🔹 Parallel BFS
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS: ";

        while (!q.empty()) {
            int size = q.size();

            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int node;

                #pragma omp critical
                {
                    if (!q.empty()) {
                        node = q.front();
                        q.pop();
                        cout << node << " ";
                    }
                }

                // Explore neighbors
                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }

    // 🔹 Parallel DFS Utility
    void parallelDFSUtil(int node, vector<bool> &visited) {
        visited[node] = true;

        #pragma omp critical
        cout << node << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                parallelDFSUtil(neighbor, visited);
            }
        }
    }

    // 🔹 Parallel DFS
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS: ";
        parallelDFSUtil(start, visited);
        cout << endl;
    }
};

int main() {
    int V = 6;
    Graph g(V);

    // Example graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}
