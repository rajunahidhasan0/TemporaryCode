#include <bits/stdc++.h>
#define P(X) cout<<"db "<<X<<endl;
#define P2(X,Y) cout<<"d2 "<<X<<" "<<Y<<endl;
#define P3(X,Y,Z) cout<<"d3 "<<X<<" "<<Y<<" "<<Z<<endl;
#define ll long long
#define rep(i,n) for(i=1;i<=n;i++)
#define FO freopen("t.txt","w",stdout);
#define MS(XX,YY) memset(XX,YY,sizeof(XX));
#define pii pair<int,int>
using namespace std;
int main()
{
    int i,j,a,b,tcs,csn=0;
    freopen("in.txt","r",stdin);
    scanf("%d",&tcs);
    while(tcs--){
        scanf("%",&);

        printf("Case %d:\n",++csn,);
    }
    return 0;
}

        ///**************How to Code************//
    #read each word separately untill newline
            std::string fl;
            std::getline(std::cin, fl); // get first line

            std::istringstream iss(fl);
            std::string word;
            while(iss >> word) {
                cout<<"|"<<word<<"|";
            }



        ///******NUMBER THEORY*******************************
///Factorial
ll ft[16];
void fact()
{
    int i=1;
    ft[0]=1;
    for(;i<13;i++)ft[i]=ft[i-1]*i;
}
///nCr
ll cr[902][902];
ll ncr(int n,int r)
{
    if(n==r)return 1;
    if(r==1)return n;
    if(cr[n][r])return cr[n][r];
    cr[n][r]=ncr(n-1,r)+ncr(n-1,r-1);
    return cr[n][r];
}

///Sive
#define N 1000009
int pr[78600],in=0;
char ap[N+3];
void siv()
{
    int i,j,sq,p;
    sq=sqrt(N)+2;
    ap[1]=1;
    for(i=2;i<sq;i++){
        if(!ap[i]){
            for(j=i*i;j<N;j+=i)ap[j]=1;
        }
    }
    for(i=2;i<N;i++){
        if(!ap[i]){
            pr[in++]=i;
        }
    }
}

    ///**Bitwise Sive
//Odd numbers only
#define on(X) (mkr[X>>6]&(1<<((X&63)>>1)))
#define mark(X) mkr[X>>6]|=(1<<((X&63)>>1))
int mkr[10000900/64],N=10000020,prm[700000],in;
void bitwsiv()
{
    int i,j,rt=sqrt(N)+1;
    for(i=3;i<=rt;i+=2){
        if(!on(i)){
            for(j=i*i;j<=N;j+=i+i){
                mark(j);
            }
        }
    }
    prm[in++]=2;
    for(i=3;i<=N;i+=2){
        if(!on(i))prm[in++]=i;
    }
    P(prm[in-1])
}

    ///***Is prime
int iprm(int n)
{
    if(n==2)return 1;
    if(!(n%2)||n<2)return 0;
    int i,sq=sqrt(n)+2;
    for(i=3;i<sq;i+=2)if(!(n%i))return 0;
    return 1;
}

    ///*** Probable prime check
// This function is called for all k trials. It returns
// false if n is composite and returns false if n is
// probably prime.
// d is an odd number such that  d*2<sup>r</sup> = n-1
// for some r >= 1
bool miillerTest(int d, int n)
{
    // Pick a random number in [2..n-2]
    // Corner cases make sure that n > 4
    int a = 2 + rand() % (n - 4);

    // Compute a^d % n
    int x = power(a, d, n);

    if (x == 1  || x == n-1)
       return true;

    // Keep squaring x while one of the following doesn't
    // happen
    // (i)   d does not reach n-1
    // (ii)  (x^2) % n is not 1
    // (iii) (x^2) % n is not n-1
    while (d != n-1)
    {
        x = (x * x) % n;
        d *= 2;

        if (x == 1)      return false;
        if (x == n-1)    return true;
    }

    // Return composite
    return false;
}

// It returns false if n is composite and returns true if n
// is probably prime.  k is an input parameter that determines
// accuracy level. Higher value of k indicates more accuracy.
bool isPrime(int n, int k)
{
    // Corner cases
    if (n <= 1 || n == 4)  return false;
    if (n <= 3) return true;

    // Find r such that n = 2^d * r + 1 for some r >= 1
    int d = n - 1;
    while (d % 2 == 0)
        d /= 2;

    // Iterate given nber of 'k' times
    for (int i = 0; i < k; i++)
         if (!miillerTest(d, n))
              return false;

    return true;
}

    ///*** Iterative Function to calculate (x^y) in O(log y) */
int power(int x, unsigned int y)
{
    int res = 1;     // Initialize result
    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = res*x;
        // n must be even now
        y = y>>1; // y = y/2
        x = x*x;  // Change x to x^2
    }
    return res;
}

    ///*** Fast Prime factorization
int minPrime[n + 1];
for (int i = 2; i * i <= n; ++i) {
    if (minPrime[i] == 0) {         //If i is prime
        for (int j = i * i; j <= n; j += i) {
            if (minPrime[j] == 0) {
                minPrime[j] = i;
            }
        }
    }
}
for (int i = 2; i <= n; ++i) {
    if (minPrime[i] == 0) {
        minPrime[i] = i;
    }
}

Now, use this ification to factorize

in O(log(N)) time.

vector<int> factorize(int n) {
    vector<int> res;
    while (n != 1) {
        res.push_back(minPrime[n]);
        n /= minPrime[n];
    }
    return res;
}

    ///*** A Lucas Theorem based solution to compute nCr % p
#include<bits/stdc++.h>
using namespace std;

// Returns nCr % p.  In this Lucas Theorem based program,
// this function is only called for n < p and r < p.
int nCrModpDP(int n, int r, int p)
{
    // The array C is going to store last row of
    // pascal triangle at the end. And last entry
    // of last row is nCr
    int C[r+1];
    memset(C, 0, sizeof(C));

    C[0] = 1; // Top row of Pascal Triangle

    // One by constructs remaining rows of Pascal
    // Triangle from top to bottom
    for (int i = 1; i <= n; i++)
    {
        // Fill entries of current row using previous
        // row values
        for (int j = min(i, r); j > 0; j--)

            // nCj = (n-1)Cj + (n-1)C(j-1);
            C[j] = (C[j] + C[j-1])%p;
    }
    return C[r];
}

// Lucas Theorem based function that returns nCr % p
// This function works like decimal to binary conversion
// recursive function.  First we compute last digits of
// n and r in base p, then recur for remaining digits
int nCrModpLucas(int n, int r, int p)
{
   // Base case
   if (r==0)
      return 1;

   // Compute last digits of n and r in base p
   int ni = n%p, ri = r%p;

   // Compute result for last digits computed above, and
   // for remaining digits.  Multiply the two results and
   // compute the result of multiplication in modulo p.
   return (nCrModpLucas(n/p, r/p, p) * // Last digits of n and r
           nCrModpDP(ni, ri, p)) % p;  // Remaining digits
}


    ///***MOD Inverse

int fr[2000009],mr[2000009];
#define M 1000000007
#define pii pair<ll,ll>
pii extnuc(ll a,ll b)
{
    if(b==0)return pii(1,0);
    pii d=extnuc(b,a%b);
    return pii(d.second,d.first-d.second*(a/b));
}

ll modi(ll n)
{
    pii d=extnuc(n,M);
    return ((d.first%M)+M)%M;
}
//Now factorial & ft inverse with MOD
void fact()
{
    ll i;
    fr[0]=mr[0]=1;
    for(i=1;i<2000007;i++){
        fr[i]=(fr[i-1]*i)%M;
        mr[i]=(mr[i-1]*modi(i))%M;
        //P(fr[i])
    }
}

ll ncr(int n,int r){
    ll r1=(((fr[n])*mr[r])%M)*mr[n-r];
    ll r2=(((fr[n])*modi(fr[r]))%M)*modi((fr[n-r]));
    r1%=M;
    r2%=M;
    if(r1!=r2){
        P2(r1,r2);

    }
    return r1;
}

    ///*** Inclusion-Exclusion from Hackerearth.
int n; // the number of sets in the set A
int result = 0; //final result, the cardinality of sum of all subsets of A
for(int b = 0; b < (1 << n); ++b)
{
     vector<int> indices;
     for(int k = 0; k < n; ++k)
     {
          if(b & (1 << k))
          {
               indices.push_back(k);
          }
     }
     int cardinality = intersectionCardinality(indices);
     if(indices.size() % 2 == 1) result += cardinality;
     else result -= cardinality;
}
cout << result << endl; //printing the final result


///    ***int Factorial ****
import java.math.BigInteger;
import java.util.Scanner;
public class Main {
	public static Scanner sc;
	public static void main(String [] arg) {

		BigInteger [] fact = new BigInteger[200];

    	fact[0] = BigInteger.ONE;

		for(int i=1;i<=150;i++) {
			fact[i] = fact[i-1].multiply(new BigInteger(i + ""));
		}

        sc = new Scanner(System.in);
        int ts = sc.nextInt();
        int n,cas=0;

        while(++cas<=ts) {
        	n = sc.nextInt();
        	System.out.println(fact[n]);
        }
	}
}


///*********Graph**************************************
    ///*** BFS
vector <int> ed[30003];
int lev[30003],ms,par[30003];
bool vs[30003];
int bfs(int s){
    int i,j,d,f,v;
    queue <int> q;
    q.push(s);
    MS(vs,0)
    //MS(lev)
    lev[s]=0;
    vs[s]=1;
    //nm=s;
    while(!q.empty()){
        f=q.front();
        q.pop();
        for(i=0;i<ed[f].size();i++){
            v=ed[f][i];
            if(!vs[v]){
                vs[v]=1;
                q.push(v);
                par[v]=f;
                lev[v]=lev[f]+1;
                //P(f<<" "<<v<<" "<<lev[v])
                //if(lev[v]>lev[nm])nm=v;
            }
        }
    }
    return nm;
}
    ///*** BFS in 2D Grid
//Grid size n*m
int lev[22][22],m,n;
bool vs[22][22];
char ar[22][22];
int dx[]={1,-1,0, 0};
int dy[]={0, 0,1,-1};

int bfs(int fx,int fy)
{
    int i,v,x,y,md=0;
    queue <pii> q;
    pii pr;
    MS(vs)
    vs[fx][fy]=1;
    lev[fx][fy]=0;
    q.push(pii(fx,fy));
    while(!q.empty()){
        pr=q.front();
        fx=pr.first;
        fy=pr.second;
        q.pop();
        for(i=0;i<4;i++){
            x=fx+dx[i];
            y=fy+dy[i];
            if(x<0||x>=n||y<0||y>=m)continue;
            if(!vs[x][y]&&ar[x][y]!='#'&&ar[x][y]!='m'){//not blocked cell(#/m blocked)
                q.push(pii(x,y));
                vs[x][y]=1;
                lev[x][y]=lev[fx][fy]+1;
                if(ar[x][y]=='a'||ar[x][y]=='b'||ar[x][y]=='c'){//a b c target cell.
                    md=max(md,lev[x][y]);
                }

            }
        }

    }
    return md;//max distance of a/b/c

}
    ///*** Union DF
vector <int> ed[20005];
int vs[20005],lev[20005],rp[20005],ex[20005],l1,l2;
int rep(int n)
{
    if(rp[n]==n)return n;
    return rp[n]=rep(rp[n]);
}
/set up/ rp[rep(b)]=rep(a);

    ///*** MST kruskal
class edge{
    public:
    int a,b,w;
    void setv(int sa,int sb,int sw){
        a=sa;
        b=sb;
        w=sw;
    }
    bool operator < (const edge& x)const {
        return w<x.w;
    }
}ed[12003];
int prr[105];
int par(int x){
    if(prr[x]==x)return x;
    return prr[x]=par(prr[x]);
}
int dis[103],ce[102],n,in;
int mnmst()
{
    int r=0,tr=0,i;
    sort(ed,ed+in);
    for(i=0;i<=102;i++)prr[i]=i;
    for(i=0;i<in;i++){
        if(par(ed[i].a)!=par(ed[i].b)){
            r+=ed[i].w;
            tr++;
            prr[par(ed[i].b)]=par(ed[i].a);
            if(tr==n)return r;
        }
    }
    return r;
}
////in main:
        scanf("%d",&n);
        in=0;
        while(1){
            scanf("%d %d %d",&a,&b,&w);
            if(!a&&!b&&!w)break;
            ed[in++].setv(a,b,w);
        }

    ///*** Topsort using DFS
//ligh oj 1034-Hit the Light Switches
int ft[10009],ttm=0;
bool vs[10009];
vector <int> ed[10009];
bool cmp(int a,int b)
{
    return  ft[a]>ft[b];
}
void dfs(int x)
{
    int i,v;
    vs[x]=1;
    for(i=0;i<ed[x].size();i++){
        v=ed[x][i];
        if(!vs[v]){
            dfs(v);
        }
    }
    ft[x]=++ttm;
}

int main()
{
    int i,j,a,b,ts,cn=0,n,cnt,ar[10009],m;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d %d",&n,&m);
        for(i=0;i<=n;i++){//we need to assign 1 to n in main array
            ar[i]=i+1;
            ed[i].clear();
        }
        for(i=0;i<m;i++){
            scanf("%d %d",&a,&b);
            ed[a].push_back(b);
        }
        MS(vs,0)
        MS(ft,0)
        ttm=0;
        for(i=1;i<=n;i++){
            if(!vs[i])dfs(i);
            //P2(i,ft[i])
        }
        sort(ar,ar+n,cmp);
        MS(vs,0)
        MS(ft,0)
        cnt=0;
        for(i=0;i<n;i++){
            if(!vs[ar[i]]){
                dfs(ar[i]);
                cnt++;
            }
        }
        printf("Case %d: %d\n",++cn,cnt);
    }
    return 0;
}

    ///*** Dijkstra
struct node{
    int id,cost;
    node(){}
    node(int nid,int ncost)
    {
        id=nid;
        cost=ncost;
    }
    bool operator < (const node&x)const{
        return  cost>x.cost;
    }
};
//U may keep a parent array to find the path.
vector <int> ed[10009],ec[10009];
int ds[10009];
void dxt(int s){
    priority_queue <node> q;
    q.push(node(s,0));
    ds[s]=0;
    node fn;
    int i,u,v;
    while(!q.empty()){
        fn=q.top();
        q.pop();
        u=fn.id;
        if(fn.cost!=ds[u])continue;
        //P(u)
        for(i=0;i<ed[u].size();i++){
            v=ed[u][i];
            if(ds[v]>ds[u]+ec[u][i]){
                ds[v]=ds[u]+ec[u][i];
                q.push(node(v,ds[v]));
            }
        }
    }
}

    ///*** Floyd Warshall: All pair shortest path
int mtx[102][102],n;//intialize with inf;
int next[102][102];//for finding path only
void wrsl()
{
    int i,j,k;
    for(i=1;i<=n;i++){//for finding path only
        for(j=1;j<=n;j++){
            next[i][j]=j;
        }
    }

    for(k=1;k<=n;k++){
        for(i=1;i<=n;i++){
            for(j=1;j<=n;j++){
                if(mtx[i][j]>mtx[i][k]+mtx[k][j]){
                    mtx[i][j]>mtx[i][k]+mtx[k][j];
                    next[i][j]=next[i][k];//for finding path only
                }
            }
        }
    }
}
//finding path using warshal, i to j
vector <int> path;
void findpath(int i,int j)
{
    path.clear();
    path.push_back(i);
    while(i!=j){
        i=next[i][j];
        path.push_back(i);
    }
}

    ///*** Bellman Ford
#define MV 2000000009000007LL
int cs[209],m,n;
ll ds[209];
bool blk[203];// blk[i] true if blk[i] part of neg cycle.
int es[40007],ed[40007];
void blmnfrd()
{
    int i,t,a,b;
    ll d;
    for(i=0;i<=n;i++){
        ds[i]=MV;
        blk[i]=false;
    }
    ds[1]=0;
    for(t=1;t<n;t++){
        for(i=0;i<m;i++){
            a=es[i];
            b=ed[i];
            d=cs[b]-cs[a];//(cost of going a to b/i'th edge)
            if(d+ds[a]<ds[b]){
                ds[b]=d+ds[a];
            }
        }
    }
    // Here we running it n times for find all nodes with neg
    // cycle. 1 time is enoguh to see if a neg cycle in graph.
    for(t=1;t<=n;t++){
        for(i=0;i<m;i++){
            a=es[i];
            b=ed[i];
            d=cs[b]-cs[a];//(cost of going a to b)
            if(d+ds[a]<ds[b]){
                blk[b]=true;//negative cycle
                ds[b]=d+ds[a];
            }
        }
    }
}
// in main: if(ds[d]==MV||ds[d]<3||blk[d])puts("?");

    ///*** Finding Articulation Point
// preset in main()
MS(dt,-1)
MS(artqp,0)
ct=1;
root=1;
tarp=0;
par[1]=-1;
farp(1);
Done preset
//
#define SZ 10002
vector <int> ed[SZ+2];
bool artqp[SZ+2];
int low[SZ+2],dt[SZ+2],par[SZ+2],root,ct=1,tarp;
void farp(int u){
    low[u]=dt[u]=ct++;
    int i,v,child=0;
    for(i=0;i<ed[u].size();i++){
        v=ed[u][i];
        if(v==par[u])continue;
        if(dt[v]==-1){
            par[v]=u;
            farp(v);
            low[u]=min(low[u],low[v]);
            child++;
            if(low[v]>=dt[u]&&root!=u)artqp[u]=true;
        }
        else {
            low[u]=min(low[u],dt[v]);
        }
    }
    if(u==root&&child>1){
        artqp[u]=true;
    }
}

    ///*** Finding Articulation Bridge
// preset in main()
MS(dt,-1)
MS(artqp,0)
ct=1;
root=1;
tarp=0;
par[1]=-1;
farp(1);
Done preset
//
#define SZ 10002
vector <int> ed[SZ+2],artqb[SZ+2];
int low[SZ+2],dt[SZ+2],par[SZ+2],root,ct=1,tarp;
void farb(int u){
    low[u]=dt[u]=ct++;
    int i,v,child=0;
    for(i=0;i<ed[u].size();i++){
        v=ed[u][i];
        if(v==par[u])continue;
        if(dt[v]==-1){
            par[v]=u;
            farb(v);
            low[u]=min(low[u],low[v]);
            child++;
            if(low[v]>dt[u]){
                artqb[u].push_back(v);
                artqb[v].push_back(u);//for undirected.
                //u to v is articulation bridge here
            }
        }
        else {
            low[u]=min(low[u],dt[v]);
        }
    }
}

    ///*** Stable Marriage Problem
//lightoj 1400 - Employment
/*Algo from wiki:
function stableMatching {
    Initialize all m ∈ M and w ∈ W to free
    while ∃ free man m who still has a woman w to propose to {
       w = first woman on m’s list to whom m has not yet proposed
       if w is free
         (m, w) become engaged
       else some pair (m',  w) already exists
         if w prefers m to m'
            m' becomes free
           (m, w) become engaged
         else
           (m', w) remain engaged
    }
}*/
#define SZ 102
int pflm[SZ][SZ];
int pflf[SZ][SZ];
int mst[SZ],curchoice[SZ],idm[SZ],n;
//mst cur selection of man.It's 0,if not selected yet.
//pflm[i][idm[i]] is the next next girl to propose by i'th man.
void stablemc()
{
    MS(mst,0)
    MS(idm,0)
    MS(curchoice,0)
    int um=1,nf;
    while(um){
        um=0;
        for(int i=1;i<=n;i++){
            if(mst[i]==0){
                um=1;
                nf=pflm[i][idm[i]++];
                if(pflf[nf][i]<pflf[nf][curchoice[nf]]){
                    mst[curchoice[nf]]=0;
                    curchoice[nf]=i;
                    mst[i]=nf;
                }
            }
        }
    }
}
//int lightoj problem 1 to n are man & n+1 to 2n are women
//here I am substructing n from id of women.
//low value of pflf[i][x](x is man's ID) means high preference.
int main()
{
    int i,j,a,b,ts,cn=0,x;
    freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d",&n);
        for(i=1;i<=n;i++){
            for(j=1;j<=n;j++){
                scanf("%d",&x);
                pflm[i][j]=x-n;
            }
        }
        for(i=1;i<=n;i++){
            for(j=1;j<=n;j++){
                scanf("%d",&x);
                pflf[i][x]=j;
            }
            pflf[i][0]=2*n+2;
        }
        stablemc();
        printf("Case %d:",++cn);
        {
            for(i=1;i<=n;i++){
                printf(" (%d %d)",i,mst[i]+n);
            }
        }
        puts("");
    }
    return 0;
}
    ///*** Strongly Connected Component
Psudocode:
1        procedure DFS(G, u):
5             color[u]  ← GREY
6             for all edges from u to v in G.adjacentEdges(u) do
7                    if color[v]=WHITE
8                            DFS(G,v)
9                    end if
10           end for
11           stk.add(source)
13           return

14      procedure DFS2(R,u, mark)
15            components[mark].add(u) //save the nodes of the new component
16            visited[u] ← true
17            for all edges from u to v in R.adjacentEdges(u) do
18                    if visited[v] ← false
19                            DFS2(R,v, mark)
20                    end if
21            end for
22            return

23        procedure findSCC(G):
24             stk ← an empty stack
25             visited[] ← null
26             color[] ← null
27             components[] ← null
28             mark=0
29             for each u in G
30                   if color[u]=WHITE
31                          DFS(G,u)
32                   end if
33             end for
34             R=reverseEdges(G)
35             while stk not empty
36                   u=stk.removeTop()
37                   if visited[u]=false
38                        mark=mark+1 //A new component found, it will be identified by ‘mark’
39                        DFS2(R,u,mark)
40                   end if
41              end for
42              return components

//Lightoj 1168 - Wishing Snake
#define SZ 1005
stack <int> st1,cmpnts[SZ];
vector <int> ed[SZ],red[SZ];
bool vis[SZ],vis2[SZ],pnd[SZ];
int ndcno[SZ],mark,n;
void dfs1(int u)
{
    int i,v;
    vis[u]=1;
    for(i=0;i<ed[u].size();i++){
        v=ed[u][i];
        if(!vis[v])dfs1(v);
    }
    st1.push(u);
}

void dfs2(int u)
{
    int i,v;
    cmpnts[mark].push(u);
    ndcno[u]=mark;
    vis2[u]=1;
    for(i=0;i<red[u].size();i++){
        v=red[u][i];
        if(!vis2[v])dfs2(v);
    }
}
int main()
{
    int i,j,a,b,ts,cn=0,pl,pe,te,f,tp,dc,v;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d",&pl);
        MS(vis,0)
        MS(vis2,0)
        MS(pnd,0)
        for(i=0;i<=1002;i++){
            ed[i].clear();
            red[i].clear();
            while(!cmpnts[i].empty())cmpnts[i].pop();
        }
        te=0;
        while(!st1.empty())st1.pop();
        while(pl--){
            scanf("%d",&pe);
            te+=pe;
            while(pe--){
                scanf("%d %d",&a,&b);
                ed[a].push_back(b);
                red[b].push_back(a);
                pnd[a]=1;
                pnd[b]=1;
            }
        }
        dfs1(0);
        f=0;
        for(i=0;i<=1000;i++){
            if(pnd[i]!=vis[i]){
                f=1;
                break;
            }
        }
        //if(f)
        mark=0;
        while(!st1.empty()){
            tp=st1.top();
            st1.pop();
            //P(tp)
            if(!vis2[tp]){
                dfs2(tp);
                mark++;
            }
        }
        //P(mark)
        for(i=0;i<mark;i++){
            dc=0;
            while(!cmpnts[i].empty()){
                tp=cmpnts[i].top();
                cmpnts[i].pop();
                for(j=0;j<ed[tp].size();j++){
                    v=ed[tp][j];
                    if(ndcno[v]!=i){
                        dc++;
                    }
                }
            }
            if(dc>1){
                f=1;
                break;
            }
        }
        if(f)printf("Case %d: NO\n",++cn);
        else printf("Case %d: YES\n",++cn);
    }
    return 0;
}

    ///*** Euler Tour
Psudocode:
tour_stack = empty stack
find_circuit(u):
     for all edges  u->v in G.adjacentEdges(v) do:
            remove u->v
            find_circuit(v)
     end for
     tour_stack.add(u)
     return

    //Lightoj  1256 - Word Puzzle :
**** I got WA in this problem
#include <bits/stdc++.h>
#define P(X) cout<<"db "<<X<<endl;
#define P2(X,Y) cout<<"d2 "<<X<<" "<<Y<<endl;
#define ll long long
#define rep(i,n) for(i=1;i<=n;i++)
#define FO freopen("t.txt","w",stdout);
#define MS(XX,YY) memset(XX,YY,sizeof(XX));
#define pii pair<int,int>
using namespace std;
int ind[30],outd[30],in,tourc[2005];
vector <string> mtx[27][27];
vector <int> ed[28];
void fulc(int u)
{
    int i,v;
    while(!ed[u].empty()){
        v=ed[u].back();
        ed[u].pop_back();
        fulc(v);
    }
    tourc[in++]=u;
}
int main()
{
    int i,j,a,b,ts,cnt=0,n,pn,cn,sp,ep,f,cyl;
    string st;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d",&n);
        MS(ind,0);
        MS(outd,0);
        for(i=0;i<26;i++){
            ed[i].clear();
            for(j=0;j<26;j++){
                mtx[i][j].clear();
            }
        }
        for(i=0;i<n;i++){
            cin>>st;
            a=st[0]-'a';
            b=st[st.size()-1]-'a';
            ind[b]++;
            outd[a]++;
            mtx[a][b].push_back(st);
            ed[a].push_back(b);
        }
        sp=ep=-1;
        f=1;
        for(i=0;i<26;i++){
            if(sp==-1&&ind[i]==outd[i]-1){
                sp=i;
            }
            else if(ep==-1&&ind[i]==outd[i]+1){
                ep=i;
            }
            else if(ind[i]!=outd[i]){
                f=0;
                break;
            }
        }
        printf("Case %d: ",++cnt);
        if(!f)puts("No");
        else{
            if(sp==-1){
                for(i=0;i<26;i++){
                    if(outd[i]){
                        sp=i;
                        break;
                    }
                }
            }
            in=0;
            fulc(sp);
            pn=tourc[in-1];
            puts("Yes");
            for(i=in-2;i>=0;i--){
                cn=tourc[i];
                cout<<mtx[pn][cn].back()<<" ";
                mtx[pn][cn].pop_back();
                pn=cn;
            }
            puts("");
        }
    }
    return 0;
}

///Graph END------------------------------------------

    ///*** String & DS************************************
    ///***segm tree tmp

int lazy[800000],tree[800000],idr[180009];
void update(int low,int high,int ulow,int uhigh,int val,int pos)
{
    if(low>high)return;
    if(lazy[pos]!=0){
        tree[pos]+=(high-low+1)*lazy[pos];
        if(low<high){
            lazy[2*pos+1]+=lazy[pos];
            lazy[2*pos+2]+=lazy[pos];
        }
        lazy[pos]=0;
    }
    if(ulow>high||uhigh<low)return;
    if(ulow<=low&&high<=uhigh){
        tree[pos]+=(high-low+1)*val;
        if(low<high){
            lazy[2*pos+1]+=val;
            lazy[2*pos+2]+=val;
        }
        return;
    }
    int mid=(low+high)/2;
    update(low,mid,ulow,uhigh,val,2*pos+1);
    update(mid+1,high,ulow,uhigh,val,2*pos+2);
    tree[pos]=tree[2*pos+1]+tree[2*pos+2];
}
int query(int low,int high,int qlow,int qhigh,int pos)
{
    int r1,r2,mid;
    if(low>high)return 0;
    if(lazy[pos]!=0){
        tree[pos]+=(high-low+1)*lazy[pos];
        if(low<high){
            lazy[2*pos+1]+=lazy[pos];
            lazy[2*pos+2]+=lazy[pos];
        }
        lazy[pos]=0;
    }
    if(qlow>high||qhigh<low)return 0;
    if(qlow<=low&&high<=qhigh){
        return tree[pos];
    }
    mid=(high+low)/2;
    r1=query(low,mid,qlow,qhigh,2*pos+1);
    r2=query(mid+1,high,qlow,qhigh,2*pos+2);
    return r1+r2;
}

//inti tmp different

int tr[300009],ar[100009];
int init(int nd,int b,int e){
    int i,ln,rn,md,lv,rv;
    if(b==e){
        tr[nd]=ar[b];
        return tr[nd];
    }
    ln=2*nd;
    rn=2*nd+1;
    md=(b+e)/2;
    lv=init(ln,b,md);
    rv=init(rn,md+1,e);
    tr[nd]=min(lv,rv);
    return tr[nd];
}

    ///***Segment Tree:
Segment tree with point update and range query. The interval is half open 0 indexed.

struct Tree {
    typedef int T;
    const T LOW = -1234567890;
    T f(T a, T b) { return max(a, b); }
    int n;
    vi s;
    Tree() {}
    Tree(int m, T def=0) { init(m, def); }
    void init(int m, T def) {
        n = 1;
        while (n < m) n *= 2;
        s.assign(n + m, def);
        s.resize(2*n, LOW);
        for (int i = n; i --> 1; )
        s[i] = f(s[i*2], s[i*2 + 1]);
    }
    void update(int pos, T val) {
        pos += n;
        s[pos] = val;
        for (pos /= 2; pos >= 1; pos /= 2)
            s[pos] = f(s[pos*2], s[pos*2 + 1]);
    }
    T query(int l, int r) { return que(1, l, r, 0, n); }
    T que(int pos, int l, int r, int lo, int hi) {
        if (r <= lo || hi <= l) return LOW;
        if (l <= lo && hi <= r) return s[pos];
        int m = (lo + hi) / 2;
        return f(que(2*pos, l, r, lo, m), que(2*pos + 1, l, r, m, hi));
    }
};

Lazy Segment Tree: Segment tree with ability to add or set values of large intervals, and compute max of intervals. Can be changed to other things.

LazyTree.cpp
static char buf[450 << 20];
void * operator new(size_t s) {
    static size_t i = sizeof buf;
    assert(s < i);
    return (void *)&buf[i -= s];
}
void operator delete(void * ) {}

const int inf = 1e9;
struct Node {
    Node *l = 0, *r = 0;
    int lo, hi, mset = inf, madd = 0, val = -inf;
    Node(int lo,int hi):lo(lo),hi(hi){} // Large interval of -inf
    Node(vi& v, int lo, int hi) : lo(lo), hi(hi) {
        if (lo + 1 < hi) {
            int mid = lo + (hi - lo)/2;
            l = new Node(v, lo, mid);
            r = new Node(v, mid, hi);
            val = max(l->val, r->val);
        }
        else val = v[lo];
    }
    int query(int L, int R) {
        if (R <= lo || hi <= L) return -inf;
        if (L <= lo && hi <= R) return val;
        push();
        return max(l->query(L, R), r->query(L, R));
    }
    void set(int L, int R, int x) {
        if (R <= lo || hi <= L) return;
        if (L <= lo && hi <= R) mset = val = x, madd = 0;
        else {
            push(), l->set(L, R, x), r->set(L, R, x);
            val = max(l->val, r->val);
        }
    }
    void add(int L, int R, int x) {
        if (R <= lo || hi <= L) return;
        if (L <= lo && hi <= R) {
            if (mset != inf) mset += x;
            else madd += x;
            val += x;
        } else {
            push(), l->add(L, R, x), r->add(L, R, x);
            val = max(l->val, r->val);
        }
    }
    void push() {
        if (!l) {
            int mid = lo + (hi - lo)/2;
            l = new Node(lo, mid); r = new Node(mid, hi);
        }
        if (mset != inf)
            l->set(lo,hi,mset), r->set(lo,hi,mset), mset = inf;
        else if (madd)
            l->add(lo,hi,madd), r->add(lo,hi,madd), madd = 0;
    }
};

Ordered Statistics Tree: A set (not multiset!) with support for finding the nth element, and finding the index of an element.


    ///*** BIT
int n=SIZE;
void update(int idx,int val)//adding value val to idx index
{
    while(idx<=n){
        bitree[idx]+=val;
        idx+=idx&(-idx);
    }
}
int query(int idx){// returns sum of 1 to idx index
    int sum=0;
    while(idx>0){
        sum+=bitree[idx];
        idx-=idx&(-idx);
    }
    return sum;
}

///KMP
char s1[100009],key[100009];
int lps[100009],lk=0,l1,l2;
void clps(char *key,int ln)
{
    int i=1,j=0;
    while(i<ln){
        if(key[i]==key[j]){
            lps[i]=++j;
            i++;
        }
        else {
            if(j)j=lps[j-1];
            else lps[i++]=0;
        }
    }
}

int src(char *txt,char *key){
int i=0,j=0;
    while(txt[i]){
        if(txt[i]==key[j]){
            j++;
            i++;
            if(j==lk){
                return i-j;
                //j=lps[j-1];
            }
        }
        else{
            if(j)j=lps[j-1];
            else i++;
        }
    }
    return -1;
}

///Hashing string

// primes 9997913 9997927 / 999721 998743 995987  / 4871 4723 971 937
char hst[1000000];
int dohs(char *s,int M=999721)
{
    ll int i,pm=937,pp=1,hv=0;
    for(i=0;s[i];i++){
        pp=(pp*pm)%M;
        hv+=s[i]*pp;
        hv%=M;
    }
    return hv;
}

/// palindrome in a range(0  ln-1) using hashing
#include <bits/stdc++.h>
#define P(X) cout<<"db "<<X<<endl;
#define P2(X,Y) cout<<"d2 "<<X<<" "<<Y<<endl;
#define ll long long
#define rep(i,n) for(i=1;i<=n;i++)
#define FO freopen("t.txt","w",stdout);
#define MS(XX,YY) memset(XX,YY,sizeof(XX));
#define pii pair<int,int>
using namespace std;
ll hst[1000000],rhst[1000000],ln,pr[1000];
// primes 9997913 9997927 / 999721 998743 995987  / 4871 4723 971 937
ll dohs(char *s,int M=999721)
{
    ll int i,pm=937,pp=1,hv=0;
    for(i=0;s[i];i++){
        pp=(pp*pm)%M;
        pr[i+1]=pp;
        hv+=s[i]*pp;
        hv%=M;
        hst[i]=hv;
    }
    return hv;
}
ll revdohs(char *s,int M=999721)
{
    ll int i,pm=937,pp=1,hv=0;
    for(i=strlen(s)-1;i>=0;i--){
        pp=(pp*pm)%M;
        hv+=s[i]*pp;
        hv%=M;
        rhst[i]=hv;
    }
    return hv;
}
#define MD 999721
pii extnuc(ll a,ll b)
{
    if(b==0)return pii(1,0);
    pii d=extnuc(b,a%b);
    return pii(d.second,d.first-d.second*(a/b));
}

ll modi(ll n)
{
    pii d=extnuc(n,MD);
    return ((d.first%MD)+MD)%MD;
}
int main()
{
    ll i,j,a,b,ts,f=1,cn=0,n,sf,sr;
    char s[1000];
    freopen("test.txt","r",stdin);
    scanf("%s",s);
    ln=strlen(s);
    dohs(s);
    revdohs(s);
    pr[0]=1;
    scanf("%lld",&n);
    for(i=0;i<n;i++){
        scanf("%lld %lld ",&a,&b);
        sf=hst[b]+MD;
        if(a)sf-=hst[a-1];
        sr=rhst[a]-rhst[b+1]+MD;
        sf*=modi(pr[a]);
        sr*=modi(pr[ln-b-1]);
        sf%=MD;
        sr%=MD;
        if(sf==sr)puts("PD");
        else puts("N PD");
    }
    return 0;
}
/// Trie form shafaetsplanet
struct node {
    bool endmark;
    node* next[26 + 1];
    node()
    {
        endmark = false;
        for (int i = 0; i < 26; i++)
            next[i] = NULL;
    }
} * root;
void insert(char* str, int len)
{
    node* curr = root;
    for (int i = 0; i < len; i++) {
        int id = str[i] - 'a';
        if (curr->next[id] == NULL)
            curr->next[id] = new node();
        curr = curr->next[id];
    }
    curr->endmark = true;
}
bool search(char* str, int len)
{
    node* curr = root;
    for (int i = 0; i < len; i++) {
        int id = str[i] - 'a';
        if (curr->next[id] == NULL)
            return false;
        curr = curr->next[id];
    }
    return curr->endmark;
}
void del(node* cur)
{
    for (int i = 0; i < 26; i++)
        if (cur->next[i])
            del(cur->next[i]);

    delete (cur);
}
int main()
{
    puts("ENTER NUMBER OF WORDS");
    root = new node();
    int num_word;
    cin >> num_word;
    for (int i = 1; i <= num_word; i++) {
        char str[50];
        scanf("%s", str);
        insert(str, strlen(str));
    }
    puts("ENTER NUMBER OF QUERY";);
    int query;
    cin >> query;
    for (int i = 1; i <= query; i++) {
        char str[50];
        scanf("%s", str);
        if (search(str, strlen(str)))
            puts("FOUND");
        else
            puts("NOT FOUND");
    }
    del(root);
    return 0;
}

    ///*** Trie LOJ DNA Prefix
int in=0;
struct tre{
    int cnt,nid[4];
}node[1000000];

int pcs(char ch){
    int id;
    switch(ch){
    case 'A':
        id=0;
        break;
    case 'C':
        id=1;
        break;
    case 'G':
        id=2;
        break;
    case 'T':
        id=3;
        break;
    }
    return id;
}
int insert(char *s){
    int i,id,cnd=0,mx=0;
    for(i=0;s[i];i++){
        id=pcs(s[i]);
        if(!node[cnd].nid[id]){
            node[cnd].nid[id]=in++;
        }
        cnd=node[cnd].nid[id];
        node[cnd].cnt++;
        mx=max(node[cnd].cnt*(i+1),mx);
    }
    return mx;
}

///this function was not submitted in oj
int search(char *s){
    int i,id,cnd=0,mx=0;
    for(i=0;s[i];i++){
        id=pcs(s[i]);
        if(!node[cnd].nid[id]){
            return false;
        }
        cnd=node[cnd].nid[id];
        node[cnd].cnt++;
    }
    return node[cnd].cnt;
    //Or return node[cnd].endmark type of something.
}

int main()
{
    int i,j,a,b,ts,cn=0,n,mx;
    char ss[100];
    freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        mx=0;
        in=1;
        MS(node,0);
        scanf("%d ",&n);
        for(i=0;i<n;i++){
            scanf("%s",ss);
            a=insert(ss);
            mx=max(mx,a);
        }
        printf("Case %d: %d\n",++cn,mx);
    }
    return 0;
}

    ///*** LCA form shafaetsplanet
//LCA using sparse table
//Complexity: O(NlgN,lgN)
#define mx 100002
int L[mx]; //Level
int P[mx][22]; //Sparce table
int T[mx]; //parent
vector<int>g[mx];
void dfs(int from,int u,int dep)
{
    T[u]=from;
    L[u]=dep;
    for(int i=0;i<(int)g[u].size();i++)
    {
        int v=g[u][i];
        if(v==from) continue;
        dfs(u,v,dep+1);
    }
}

int lca_query(int N, int p, int q) //N=Total node
  {
      int tmp, log, i;

      if (L[p] < L[q])
          tmp = p, p = q, q = tmp;

        log=1;
      while(1) {
        int next=log+1;
        if((1<<next)>L[p])break;
        log++;

      }

        for (i = log; i >= 0; i--)
          if (L[p] - (1 << i) >= L[q])
              p = P[p][i];

      if (p == q)
          return p;

        for (i = log; i >= 0; i--)
          if (P[p][i] != -1 && P[p][i] != P[q][i])
              p = P[p][i], q = P[q][i];

      return T[p];
  }

void lca_init(int N)
  {
      memset (P,-1,sizeof(P));//All parent -1 at first
      int i, j;
       for (i = 0; i < N; i++)
          P[i][0] = T[i];

      for (j = 1; 1 << j < N; j++)
         for (i = 0; i < N; i++)
             if (P[i][j - 1] != -1)
                 P[i][j] = P[P[i][j - 1]][j - 1];
 }

int main(void) {
	g[0].pb(1);
	g[0].pb(2);
	g[2].pb(3);
	g[2].pb(4);
	dfs(0, 0, 0);
	lca_init(5);
	printf( "%d\n", lca_query(5,3,4) );
	return 0;
}

    ///**LCA LOJ1101 - A Secret Mission
#define MX 50009
using namespace std;
int par[MX],w,m,lev[MX],spt[MX][26],n;
int mxct[MX][26],ptc[MX];
vector <int> ed[MX],ec[MX];
struct edge{
    int a, b, w;
    void sedge(int sa,int sb,int sw){
        a=sa;
        b=sb;
        w=sw;
    }
    bool operator <(const edge&x)const{
        return w<x.w;
    }
}ieg[100009];
int fpr(int x)
{
    if(par[x]==x)return x;
    else return par[x]=fpr(par[x]);
}
int dfs(int prr,int nd){
    int v;
    for(int i=0;i<ed[nd].size();i++){
        v=ed[nd][i];
        if(v==prr)continue;
        lev[v]=lev[nd]+1;
        par[v]=nd;
        ptc[v]=ec[nd][i];
        dfs(nd,v);
    }

}
void lca_setup()
{
    int i,j,st;
    for(i=1;i<=n;i++){
        spt[i][0]=par[i];
        mxct[i][0]=ptc[i];
    }
    for(j=1;(1<<j)<=n;j++){
        for(i=1;i<=n;i++){
            if(spt[i][j-1]!=-1){
                spt[i][j]=spt[spt[i][j-1]][j-1];
                mxct[i][j]=max(mxct[spt[i][j-1]][j-1],mxct[i][j-1]);
            }
        }
    }
}
int lcaqry(int p,int q)
{
    if(lev[q]>lev[p])swap(p,q);
    int i,j,lg=0,cs=0;
    while(1){
        if((1<<lg)>lev[p])break;
        lg++;
    }
    //P2(lev[p],lev[q])
    for(i=lg;i>=0;i--){
        if((1<<i)+lev[q]<=lev[p]){
            cs=max(cs,mxct[p][i]);
            p=spt[p][i];
        }
    }
    if(p==q){
        return cs;
    }
    for(i=lg;i>=0;i--){
        if(spt[p][i]!=-1&&spt[p][i]!=spt[q][i]){
            cs=max(cs,mxct[p][i]);
            cs=max(cs,mxct[q][i]);
            p=spt[p][i];
            q=spt[q][i];

        }
    }
    return max(cs,max(ptc[p],ptc[q]));

}
int main()
{
    int i,j,a,b,ts,cn=0,q;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d %d",&n,&m);
        for(i=0;i<=n;i++){
            par[i]=i;
            ed[i].clear();
            ec[i].clear();
        }
        for(i=0;i<m;i++){
            scanf("%d %d %d",&a,&b,&w);
            ieg[i].sedge(a,b,w);
        }
        sort(ieg,ieg+m);
        for(i=0;i<m;i++){
            if(fpr(ieg[i].a)!=fpr(ieg[i].b)){
                par[fpr(ieg[i].a)]=fpr(ieg[i].b);
                ed[ieg[i].a].push_back((ieg[i].b));
                ed[ieg[i].b].push_back((ieg[i].a));
                ec[ieg[i].b].push_back((ieg[i].w));
                ec[ieg[i].a].push_back((ieg[i].w));
            }
        }
        lev[1]=0;
        par[1]=-1;
        MS(spt,-1)
        MS(mxct,0)
        dfs(-1,1);
        printf("Case %d:\n",++cn);
        lca_setup();
        scanf("%d",&q);
        while(q--){
            scanf("%d %d",&a,&b);
            printf("%d\n",lcaqry(a,b));
        }
    }
    return 0;
}

   ///***Suffix Array+LCP: LOJ 1314 - Names for Babies
    // distinct substr of S in a,b lenth range
int sfa[10009],pos[10009],tmp[10009],lcp[10009],gap=1,ln;
char ss[10009];
bool scmp(int a,int b)
{
    if(pos[a]!=pos[b])return pos[a]<pos[b];
    a+=gap;
    b+=gap;
    return (a<ln&&b<ln)?pos[a]<pos[b]:a>b;

}
void buildsa()
{
    int i,j;
    ln=strlen(ss);
    for(i=0;i<=ln;i++){
        sfa[i]=i;
        pos[i]=ss[i];
    }
    for(gap=1;;gap*=2){
        sort(sfa,sfa+ln,scmp);
        for(i=0;i<ln;i++){
            tmp[i+1]=tmp[i]+scmp(sfa[i],sfa[i+1]);
        }
        for(i=0;i<ln;i++)pos[sfa[i]]=tmp[i];
        if(tmp[i]==ln-1)break;
    }
}
void buildlcp()
{
    int i,j,k;
    for(i=0,k=0;i<ln;i++){
        if(pos[i]==ln-1)continue;
        for(j=sfa[pos[i]+1];ss[i+k]==ss[j+k];)k++;
        lcp[pos[i]]=k;
        if(k)k--;
    }
}
int main()
{
    int i,j,a,b,ts,cn=0,n,ds=0,cr;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        ds=0;
        scanf(" %s",ss);
        scanf("%d %d",&a,&b);
        buildsa();
        buildlcp();
        a--;
        for(i=0;i<ln;i++){
            cr=min(b,ln-sfa[i]);
            if(i&&cr>a){
                cr-=max(a,lcp[i-1]);
            }
            else {
                if(a>0)cr-=a;
            }
            if(cr>0)ds+=cr;
        }
        printf("Case %d: %d\n",++cn,ds);
    }
    return 0;
}

    ///*****MAX Flow & BPM********************************
/*
    Author       :  Jan
    Problem Name : 1184 - Marriage Media
    Algorithm    : bipartiteMatching
    Complexity   :
*/
int caseno, cases, m, n;
int adj[55][55], deg[55];

struct info {
    int h, a, d;
}M[55], N[55];

int Left[55], Right[55];
bool visited[55];
bool bpm( int u ) {
    for(int i = 0, v; i < deg[u]; i++) {
        v = adj[u][i];
        if( visited[v] ) continue;
        visited[v] = true;
        if( Right[v] == -1 || bpm( Right[v] ) ) {
            Right[v] = u, Left[u] = v;
            return true;
        }
    }
    return false;
}
int bipartiteMatching() { // Returns Maximum Matching
    memset ( Left, -1, sizeof( Left ) );
    memset ( Right, -1, sizeof( Right ) );
    int i, cnt=0;
    for( i = 0; i < m; i++ ) {
        memset( visited, 0, sizeof( visited ) );
        if( bpm( i ) ) cnt++;
    }
    return cnt;
}

int main()
{
    freopen("d.in", "r", stdin);
    freopen("d.ans", "w", stdout);

    double cl = clock();

    scanf("%d", &cases);
    while( cases-- ) {
        scanf("%d %d", &m, &n);

        for( int i = 0; i < m; i++ ) scanf("%d %d %d", &M[i].h, &M[i].a, &M[i].d);
        for( int i = 0; i < n; i++ ) scanf("%d %d %d", &N[i].h, &N[i].a, &N[i].d);

        for( int i = 0; i < m; i++ ) {
            deg[i] = 0;
            for( int j = 0; j < n; j++ ) if( M[i].d == N[j].d ) {
                if( abs( M[i].h - N[j].h ) <= 12 && abs( M[i].a - N[j].a ) <= 5 ) {
                    adj[i][deg[i]++] = j;
                }
            }
        }
        printf("Case %d: %d\n", ++caseno, bipartiteMatching());
    }

    cl = clock() - cl;
    fprintf( stderr, "%lf\n", cl / 1000 );
    return 0;
}

/// FLOW dinic
/// 1153 - Internet Bandwidth  by Jami Sir

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define myabs(a) ((a)<0?(-(a)):(a))
#define pi acos(-1.0)
#define inf (1<<25)
#define CLR(a) memset(a,0,sizeof(a))
#define SET(a) memset(a,-1,sizeof(a))
#define pb push_back
#define all(a) a.begin(),a.end()
#define ff first
#define ss second
#define eps 1e-9
#define root 1
#define lft 2*idx
#define rgt 2*idx+1
#define cllft lft,st,mid
#define clrgt rgt,mid+1,ed
#define i64 long long
#define MAX 109
#define MOD 1000000007

typedef pair<int,int> pii;

int deg[MAX], adj[MAX][MAX], cap[MAX][MAX],  q[100000];
bool fl[MAX][MAX];

int dinic( int n,int s,int t ) {
    int prev[MAX], u, v, i, z, flow = 0, qh, qt, inc;
    while(1) {
        memset( prev, -1, sizeof( prev ) );
        qh = qt = 0;
        prev[s] = -2;
        q[qt++] = s;
        while( qt != qh && prev[t] == -1 ) {
            u = q[qh++];
            for(i = 0; i < deg[u]; i++) {
                v = adj[u][i];
                if( prev[v] == -1 && cap[u][v] ) {
                    prev[v] = u;
                    q[qt++] = v;
                }
            }
        }
        if(prev[t] == -1) break;
        for(z = 1; z <= n; z++) if( prev[z] !=- 1 && cap[z][t] ) {
            inc = cap[z][t];
            for( v = z, u = prev[v]; u >= 0; v = u, u=prev[v]) inc = min( inc, cap[u][v] );
            if( !inc ) continue;
            cap[z][t] -= inc;
            cap[t][z] += inc;
            for(v=z, u = prev[v]; u >= 0; v = u, u = prev[v]) {
                cap[u][v] -= inc;
                cap[v][u] += inc;
            }
            flow += inc;
        }
    }
    return flow;
}


int main(){
    //freopen("in.txt","r",stdin);
    int cs,t=1,n,i,u,v,w,m,s,d;
    scanf("%d",&cs);
    while(cs--){
        scanf("%d",&n);
        CLR(cap);
        CLR(deg);
        CLR(fl);
        scanf("%d %d %d",&s,&d,&m);
        while(m--){
            scanf("%d %d %d",&u,&v,&w);
            if(!fl[u][v] && w){
                adj[u][deg[u]++]=v;
                adj[v][deg[v]++]=u;
                fl[u][v]=fl[v][u]=1;
            }
            cap[u][v]+=w;
            cap[v][u]+=w;
        }
        printf("Case %d: %d\n",t++,dinic(n,s,d));
    }
    return 0;
}


    ///1153 - Internet Bandwidth my code:
vector <int> ed[102];
int flw[102][102],cap[102][102],par[102],rcp[102];
char vis[101];
int s,t,d;
int dfs()
{
    queue <int> q;
    int v,f,i;
    q.push(s);
    MS(vis,0);
    vis[s]=1;
    rcp[s]=1E7;
    while(!q.empty()){
        f=q.front();
        q.pop();
        for(i=0;i<ed[f].size();i++){
            v=ed[f][i];
            if(!vis[v]&&(cap[f][v]-flw[f][v]>0)){
                vis[v]=1;
                par[v]=f;
                rcp[v]=min(cap[f][v]-flw[f][v],rcp[f]);
                if(v==d){
                    return 1;
                }
                q.push(v);
            }
        }
    }
    return 0;
}

int main()
{
    int i,j,a,b,ts,cn=0,n,m,w,cp,cfl,tf;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d %d %d %d",&n,&s,&d,&m);
        for(i=0;i<=n;i++){
            ed[i].clear();
        }
        MS(cap,0)
        for(i=0;i<m;i++){
            scanf("%d %d %d",&a,&b,&w);
            if(w==0)continue;
            if(!cap[a][b]){
                ed[a].push_back(b);
                ed[b].push_back(a);
            }
            cap[a][b]+=w;
            cap[b][a]+=w;
        }
        tf=0;
        MS(flw,0);
        while(dfs()){
            cp=d;
            cfl=rcp[d];
            tf+=cfl;
            //P(tf)
            do{
                flw[par[cp]][cp]+=cfl;
                flw[cp][par[cp]]-=cfl;
                cp=par[cp];
            }while(cp!=s);
        }
        printf("Case %d: %d\n",++cn,tf);
    }
    return 0;
}

 ///1155 - Power Transmission :
int t,mtx[205][205],flw[205][205];
vector <int> ed[205];
int par[205],mfl[205];
char vis[205];
int bfs()
{
    int v,f,i;
    queue<int> q;
    q.push(1);
    MS(vis,0);
    vis[1]=1;
    mfl[1]=1e8;
    while(!q.empty()){
        f=q.front();
        q.pop();
        for(i=0;i<ed[f].size();i++){
            v=ed[f][i];
            if(!vis[v]&&(mtx[f][v]-flw[f][v]>0)){
                vis[v]=1;
                par[v]=f;
                mfl[v]=min((mtx[f][v]-flw[f][v]),mfl[f]);
                if(v==t)return 1;
                q.push(v);
            }
        }
    }
    return 0;
}

int main()
{
    int i,j,a,b,ts,cn=0,n,v,m,w,d,id,tf,flv,cp;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d",&n);
        MS(mtx,0);
        MS(flw,0);
        for(i=0;i<=n;i++){
            ed[i].clear();
        }
        for(i=1;i<=n;i++){
            scanf("%d",&v);
            mtx[2*i][2*i+1]=v;
            ed[i*2].push_back(2*i+1);
            ed[i*2+1].push_back(2*i);
        }
        scanf("%d",&m);
        for(i=1;i<=m;i++){
            scanf("%d %d %d",&a,&b,&w);
            a=2*a+1;
            b=2*b;
            if(!mtx[a][b]){
                ed[a].push_back(b);
                ed[b].push_back(a);
            }

            mtx[a][b]+=w;

        }
        scanf("%d %d",&b,&d);
        t=2*n+2;
        for(i=1;i<=b;i++){
            scanf("%d",&id);
            id=id*2;
            ed[1].push_back(id);
            mtx[1][id]=1e8;
        }
        for(i=1;i<=d;i++){
            scanf("%d",&id);
            id=id*2+1;
            ed[id].push_back(t);
            mtx[id][t]=1e8;
        }
        tf=0;
        while(bfs()){
            flv=mfl[t];
            tf+=flv;
            cp=t;
            do{
                flw[par[cp]][cp]+=flv;
                flw[cp][par[cp]]-=flv;
                cp=par[cp];
            }while(cp!=1);
        }

        printf("Case %d: %d\n",++cn,tf);
    }
    return 0;
}

    ///***DP****
    ///***Derangement:
    d(n)=(n−1)∗(d(n−1)+d(n−2))
    Base Case: d(1)=0,d(2)=1
    ///Subsetsum: LOJ Coin Change3
    int mkr[100009],coin[111],tms[111],m,lm,md;
int main()
{
    int i,j,a,b,ts,cn=0,n;
    //freopen("test.txt","r",stdin);
    scanf("%d",&ts);
    while(ts--){
        scanf("%d %d",&n,&m);
        MS(mkr,-1);
        mkr[0]=0;
        for(i=0;i<n;i++){
            scanf("%d",&coin[i]);
        }
        for(i=0;i<n;i++){
            scanf("%d",&tms[i]);
        }
        lm=0;
        for(i=0;i<n;i++){
            for(j=coin[i];j<=m;j++){
                if(mkr[j-coin[i]]!=-1&&mkr[j-coin[i]]<tms[i]&&mkr[j]==-1){
                    mkr[j]=mkr[j-coin[i]]+1;
                    mkr[j-coin[i]]=0;
                    lm=j;
                }
                else if(mkr[j-coin[i]]>0){
                    mkr[j-coin[i]]=0;
                }
            }

            for(j-=coin[i];j<=m;j++){
                if(mkr[j]>0){
                    mkr[j]=0;
                }
            }

            mkr[lm]=0;
        }
        md=0;
        for(i=1;i<=m;i++){
            //if(mkr[i]>0)P(mkr[i])
            if(mkr[i]>=0)md++;
        }

        printf("Case %d: %d\n",++cn,md);
    }
    return 0;
}

    ///****Misalc******************************************
///***Bit
int Set(int N,int pos){return N=N | (1<<pos);}
int reset(int N,int pos){return N= N & ~(1<<pos);}
bool check(int N,int pos){return (bool)(N & (1<<pos));}

#define bset(N,P) N=N|(1<<P);
#define bchk(N,P) ((bool)(N&(1<<P)))
#define brst(N,P) N=N&~(1<<P);

    ///*** Josephus Problem
int main()
{
    int i,j,a,b,ts,cn=0;
    ll n,k,s;
    scanf("%d",&ts);
    while(ts--){
        scanf("%lld %lld",&n,&k);
        s=1;
        for(i=2;i<=n;i++){
            s+=k;
            s%=i;
            if(!s)s=i;
        }
        printf("Case %d: %lld\n",++cn,s);
    }
    return 0;
}
    ///*** Matrix EXPO
ll M;
struct mtx{
    ll mr[6][6],r,c;
    mtx(){
        MS(mr,0);
        r=6,c=6;
        //pnt();
    }
    void pnt()
    {
        //This function is only for Debuging
        P2(r,c)
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                printf("%lld ",mr[i][j]);
            }
            puts("");
        }
        puts("");
    }
    mtx operator *(const mtx&x)const{
        mtx tmp;
        //P2(r,c)
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                tmp.mr[i][j]=0;
                for(int k=0;k<c;k++){
                    tmp.mr[i][j]+=mr[i][k]*x.mr[k][j];
                    tmp.mr[i][j]%=M;
                }
            }
        }
        tmp.c=x.c;
        return tmp;
    }
};
mtx mtp,am,ans;
mtx mp,idt;
mtx mpw(ll p){
    if(p==0)return idt;
    mtx re=mpw(p/2);
    //re.pnt();
    re=re*re;
    //re.pnt();

    if(p%2)re=re*mtp;
    return re;
}
int main()
{
    ll i,j,a,b,ts,cn=0,n,q,x;
    //freopen("test.txt","r",stdin);
    for(i=0;i<6;i++)idt.mr[i][i]=1;
    scanf("%lld",&ts);
    while(ts--){

        am.c=1;
        scanf("%lld %lld %lld",&mtp.mr[0][0],&mtp.mr[0][1],&mtp.mr[0][5]);
        scanf("%lld %lld %lld",&mtp.mr[3][3],&mtp.mr[3][4],&mtp.mr[3][2]);
        scanf("%lld %lld %lld",&am.mr[2][0],&am.mr[1][0],&am.mr[0][0]);
        scanf("%lld %lld %lld",&am.mr[5][0],&am.mr[4][0],&am.mr[3][0]);
        mtp.mr[1][0]=mtp.mr[2][1]=mtp.mr[4][3]=mtp.mr[5][4]=1;
        //mtp.pnt();
        ans.c=am.c=1;
        //am.pnt();
        printf("Case %lld:\n",++cn);
        scanf("%lld",&M);
        scanf("%lld",&q);
        for(i=0;i<6;i++)am.mr[i][0]%=M;
        while(q--){
            scanf("%lld",&x);
            //P(x)
            if(x<3){
                printf("%lld %lld\n",am.mr[2-x][0],am.mr[5-x][0]);
            }
            else {
                x-=2;
                ans=mpw(x);
                ans=ans*am;
                printf("%lld %lld\n",ans.mr[0][0],ans.mr[3][0]);
            }
        }
    }
    return 0;
}
    ***///Matrix CM
#define MAX 100
int row[MAX], col[MAX];
int dp[MAX][MAX];
bool visited[MAX][MAX];
int f(int beg,int end)
{
	if(beg>=end)return 0;
	if(visited[beg][end])return dp[beg][end];
	int ans=1<<30; //inf
	for(int mid=beg; mid<end;mid++)
	{
		int opr_left = f(beg, mid); //opr = multiplication operation
		int opr_right = f(mid+1, end);
		int opr_to_multiply_left_and_right = row[beg]*col[mid]*col[end];
		int total = opr_left + opr_right + opr_to_multiply_left_and_right;
		ans = min(ans, total);
	}
	visited[beg][end] = 1;
	dp[beg][end] = ans;
	return dp[beg][end];
}

int main()
{
	int n;
	cin>>n;
	rep(i,n)cin>>row[i]>>col[i];
	cout<<f(0,n-1)<<endl;
}

///bs
#include <bits/stdc++.h>
#define P(X) cout<<"db "<<X<<endl;
#define ll long long
#define rep(i,n) for(i=1;i<=n;i++)
#define FO freopen("t","w",stdout);
using namespace std;
struct st{
    int a,b;
    bool operator < (const st&x)const {
        return a<x.a;
    }
}s1[555],s2;
int x;
int bns(st n)
{
    int a=0,b=x,m;
    for(;a<=b;){
        m=(a+b)/2;
        if(s1[m].a<n.a)a=m+1;
        else if(s1[m].a>n.a)b=m-1;
        else return m;
    }
    return -1;
}
int main()
{
    int i,j,a,b,ts,cn=0,n;
    freopen("test.txt","r",stdin);
    scanf("%d",&x);
    for(i=0;i<x;i++){
        scanf("%d",&s1[i].a);
        s1[i].b=i;
    }
    sort(s1,s1+x);
    while(scanf("%d",&s2.a)==1) {
        n=bns(s2);
        if(n!=-1)printf("Case %d: %d %d\n",++cn,s1[n].b,n);
        else puts("Not found!");
    }
    return 0;
}
    ///THE END


