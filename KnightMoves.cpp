#include<bits/stdc++.h>
using namespace std;
int m, n;
int vis[50][50];
int r[10]={2, 2, -1, +1, -2, -2, -1, +1};
int c[10]={-1, +1, 2, 2, -1, +1, -2, -2};
bool valid(int u, int v)
{
    if( u<m && u>=0 && v<n && v>=0 && vis[u][v]!=1) return true;

    return false;
}

int KnightM(int nodeR, int nodeC, int cnt)
{
    if(cnt==m*n){
       // cout<<nodeR<<" "<<nodeC<<endl;
        return 1;
    }
    vis[nodeR][nodeC]=1;
    for(int i=0 ; i<8 ;i++)
    {

        int x=nodeR+r[i], y=nodeC+c[i];
        if(valid(x, y)==true)
        {
            vis[x][y]=1;
            if(KnightM(x, y, cnt+1)==1){
                cout<<x<<" "<<y<<endl;
                return 1;
            }
            else{
                vis[x][y] = -1;
            }
        }
    }
    vis[nodeR][nodeC]=-1;
    return 0;

}

int main()
{
    int test, tc=0;
    scanf("%d", &test);
    while(test--)
    {
        memset(vis, -1, sizeof(vis));

        scanf("%d %d", &m, &n);

        if(KnightM(0, 0, 1)==1){
            cout<<0<<" "<<0<<endl;
            cout<<"YES"<<endl;
        }
        else cout<<"NO"<<endl;
    }
    return 0;
}
