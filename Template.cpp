                                ///Template For ICPC 10-11-2018///
###Team:RU_Recruits(Nahid,Nafis,Zihad)

                        /*{1}# Number Theory related code*/
///For Begining Part

#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define MOD 1000000007LL
#define MS(ARRAY,VALUE) memset(ARRAY,VALUE,sizeof(ARRAY))
#define Fin freopen("input.txt","r",stdin)
#define Fout freopen("output.txt","w",stdout)
#define rep(i,a,b) for(i=a;i<=b;i++)
#define EPS 0.00000001
#define INF INT_MAX
#define PI 2*acos(0.0)
#define c1(XX) cout<<XX<<endl
#define c2(XX,YY) cout<<XX<<" "<<YY<<endl
#define c3(XX,YY,ZZ) cout<<XX<<" "<<YY<<" "<<ZZ<<endl
#define set(XX,POS) XX|(1<<POS)
#define reset(XX,POS) XX&(~(1<<POS))
#define check(XX,POS) (bool)(XX&(1<<POS))
#define toggle(XX,POS) (XX^(1<<POS))
#define SORT(v) sort(v.begin(),v.end())
#define REVERSE(V) reverse(v.begin(),v.end())
#define VALID(X,Y,R,C) X>=0 && X<R && Y>=0 && Y<C
#define SIZE(ARRAY) sizeof(ARRAY)/sizeof(ARRAY[0])
#define FAST ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#define RT printf("Run Time : %0.3lf seconds\n", clock()/(CLOCKS_PER_SEC*1.0))

template <class X> X lcm(X a, X b)
{
    return (a*b)/gcd(a,b);
}
struct str{
    int a;
    int b;
    setf(int aa, int bb){
        a=aa; b=bb;
    }
};
int level[1000];
bool visited[1000];
int parent[1000];
int vis[100005];
int dp[100005];
int a[100005];
vector<int>adj[100005];
map<int, int>mp;
queue<int>Q;
stack<int>st;
pair<int, int>pii;

int main()
{
    int test, tc=0;
    scanf("%d", &test);
    while(test--){
        int n, m;
        scanf("%d", &n);
        for(int i=0; i<n ;i++){
            scanf("%d", &a[i]);
        }
        printf("Case %d: %d\n", ++tc, ans);
    }
    return 0;
}

                            ///BigINT Calculation
struct Bigint {
    string a;
    int sign;
    Bigint() {}
    Bigint( string b ) { (*this) = b; }
    int size() {
        return a.size();
    }
    Bigint inverseSign() { // changes the sign
        sign *= -1;
        return (*this);
    }
    Bigint normalize( int newSign ) { // removes leading 0, fixes sign
        for( int i = a.size() - 1; i > 0 && a[i] == '0'; i-- )
            a.erase(a.begin() + i);
        sign = ( a.size() == 1 && a[0] == '0' ) ? 1 : newSign;
        return (*this);
    }
    void operator = ( string b ) { // assigns a string to Bigint
        a = b[0] == '-' ? b.substr(1) : b;
        reverse( a.begin(), a.end() );
        this->normalize( b[0] == '-' ? -1 : 1 );
    }
    bool operator < ( const Bigint &b ) const { // less than operator
        if( sign != b.sign ) return sign < b.sign;
        if( a.size() != b.a.size() )
            return sign == 1 ? a.size() < b.a.size() : a.size() > b.a.size();
        for( int i = a.size() - 1; i >= 0; i-- ) if( a[i] != b.a[i] )
            return sign == 1 ? a[i] < b.a[i] : a[i] > b.a[i];
        return false;
    }
    bool operator == ( const Bigint &b ) const { // operator for equality
        return a == b.a && sign == b.sign;
    }
    Bigint operator + ( Bigint b ) { // addition operator overloading
        if( sign != b.sign ) return (*this) - b.inverseSign();
        Bigint c;
        for(int i = 0, carry = 0; i<a.size() || i<b.size() || carry; i++ ) {
            carry+=(i<a.size() ? a[i]-48 : 0)+(i<b.a.size() ? b.a[i]-48 : 0);
            c.a += (carry % 10 + 48);
            carry /= 10;
        }
        return c.normalize(sign);
    }
    Bigint operator - ( Bigint b ) { // subtraction operator overloading
        if( sign != b.sign ) return (*this) + b.inverseSign();
        int s = sign; sign = b.sign = 1;
        if( (*this) < b ) return ((b - (*this)).inverseSign()).normalize(-s);
        Bigint c;
        for( int i = 0, borrow = 0; i < a.size(); i++ ) {
            borrow = a[i] - borrow - (i < b.size() ? b.a[i] : 48);
            c.a += borrow >= 0 ? borrow + 48 : borrow + 58;
            borrow = borrow >= 0 ? 0 : 1;
        }
        return c.normalize(s);
    }
    Bigint operator * ( Bigint b ) { // multiplication operator overloading
        Bigint c("0");
        for( int i = 0, k = a[i] - 48; i < a.size(); i++, k = a[i] - 48 ) {
            while(k--) c = c + b; // ith digit is k, so, we add k times
            b.a.insert(b.a.begin(), '0'); // multiplied by 10
        }
        return c.normalize(sign * b.sign);
    }
    Bigint operator / ( Bigint b ) { // division operator overloading
        if( b.size() == 1 && b.a[0] == '0' ) b.a[0] /= ( b.a[0] - 48 );
        Bigint c("0"), d;
        for( int j = 0; j < a.size(); j++ ) d.a += "0";
        int dSign = sign * b.sign; b.sign = 1;
        for( int i = a.size() - 1; i >= 0; i-- ) {
            c.a.insert( c.a.begin(), '0');
            c = c + a.substr( i, 1 );
            while( !( c < b ) ) c = c - b, d.a[i]++;
        }
        return d.normalize(dSign);
    }
    Bigint operator % ( Bigint b ) { // modulo operator overloading
        if( b.size() == 1 && b.a[0] == '0' ) b.a[0] /= ( b.a[0] - 48 );
        Bigint c("0");
        b.sign = 1;
        for( int i = a.size() - 1; i >= 0; i-- ) {
            c.a.insert( c.a.begin(), '0');
            c = c + a.substr( i, 1 );
            while( !( c < b ) ) c = c - b;
        }
        return c.normalize(sign);
    }
    void print() {  // output method
        if( sign == -1 ) putchar('-');
        for( int i = a.size() - 1; i >= 0; i-- ) putchar(a[i]);
    }
};

int main() {
    Bigint a, b, c;
    string input;
    cin >> input;
    a = input;
    cin >> input;
    b = input;
    c = a + b;  c.print();  puts("");
    c = a - b;  c.print();  puts("");
    c = a * b;  c.print();  puts("");
    c = a / b;  c.print();  puts("");
    c = a % b;  c.print();  puts("");
    if( a == b ) puts("equal");
    if( a < b ) puts("a is smaller than b");
    return 0;
}
*********************binary search****************************
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

                /*For BigMod*/
//cout<<bigmod(333333336LL,8LL,MOD)<<endl; a^b%M
LL bigmod(LL a, LL b, LL M)
{
    if(b == 0) return 1 % M;
    LL x = bigmod(a, b/2, M);
    x = (x*x)%M;
    if(b%2 == 1) x = (x*a)%M;
    return x;
}
                /*For GCD && LCM*/
//cout<<gcd(a,b)<<endl; cout<<lcm(a,b)<<endl;
template <class X>  X gcd(X a, X b) //X==int
{
    if(b==0) return a;
    if(a%b == 0) return b;
    return gcd(b, a%b);
}
template <class X> X lcm(X a, X b)
{
    return (a*b)/gcd(a,b);
}
                /***Extended GCD*/
///EQ:    ax + by = gcd(a, b)
///Input: a = 30, b = 20     Output: gcd = 10 , x = 1, y = -1
///FunctionCall:    gcdExtended(a, b, &x, &y);
int gcdExtended(int a, int b, int *x, int *y){
    if (a == 0){*x = 0; *y = 1; return b;}
    int x1, y1;
    int gcd = gcdExtended(b%a, a, &x1, &y1);
    *x = y1 - (b/a) * x1;
    *y = x1;
    return gcd;
}
                /*For Factorial*/
///Generate Factorial & modInvers for 1 to n numbers
ll fr[2000007], mr[2000007], mod=1000000009;
void fact(){
    ll i;
    fr[0]=1; mr[0]=1;
    for(int i=1 ; i< 2000007 ; i++){
        fr[i]=(fr[i-1]*i)%mod;
        mr[i]=(mr[i-1]*modi(i))%mod;
    }
}
///type 2
void fact(int n){
    ft[0]=1;
    for(int i=1; i<n; i++) ft[i]=ft[i-1]*i;
}

                    /*Modular Inverse*/
//(a/b)%M <--->((a%M)*((b^-1)%M))%M    *modii(b)==(b^-1)%M
// (a/b)%M <---> cout<<((a%M)* modi(b) )%M<<endl;    |-> a must be divisible by b
#define ll long long
int fr[2000009], mr[2000009];
#define M 1000000009
#define pii pair<int, int>
pii extunc(ll a, ll b){
    if(b==0)return pii(1,0);
    pii d=extunc(b, a%b);
    return pii(d.second,d.first-d.second*(a/b));
}
ll modi(ll n){
    pii d=extunc(n, M);
    return ((d.first%M)+M)%M;
}

            /*nCr*/
LL nCr(LL n, LL r, LL p)
{
	return (fact[n]%p * modi((fact[r] * fact[n-r])%p,p)%p)%p;
}

            /*Primality Test*/
bool is_prime(ll n)
{
    if(n<=1) return false;
    if(n<=3) return true;
    if(n%2 == 0 || n%3 == 0) return false;
    for(ll i=5;i*i<=n;i+=6)
        if(n%i == 0 || n%(i+2) == 0) return false;
    return true;
}
                /*For Bitwise Sieve*/
// Bsieve(100000100);   cout<<nprime<<endl;
int prime[10000000];       // 10^7
int mark[100000100/32];     //10^8/32
int nprime=0;           //nprime==number of prie from 2 to 100000100
void Bsieve(int n)
{
	int i,j,limit = sqrt(n)+2;

	mark[1/32]=Set(mark[1/32], 1%32);

	for(i=4; i<=n; i+=2) mark[i/32]=Set(mark[i/32], i%32);
	prime[++nprime] = 2;

	for(i=3; i<=n; i+=2)
	{
		if(check(mark[i/32],i%32)==0)
		{
			prime[++nprime] = i;
			if(i <= limit)
			{
				for(j=i*i; j<=n; j+=2*i)
				{
					mark[j/32]=Set(mark[j/32], j%32);
				}
			}
		}
	}
	return;
}

                  /*For General Sieve*/
// n=10000100; sieve(50000); cout<<prime[i]<<endl;
LL prime[10000000];
bool mark[10000110];    //*int mark[10000110]
LL nprime=0;
void sieve(LL n){
    LL i,j,limit = sqrt(n)+2;
    mark[1] = 1;
    for(i=2;i<=n;i+=2) mark[i]=1; //*mark[i]=2;
    prime[++nprime]=2;
    for(i=3;i<=n;i+=2){
        if(!mark[i]){
            prime[++nprime]=i;
            //*mark[i]=i;
            if(i<=limit){
                for(j=i*i;j<=n;j+=i*2)
                    mark[j] = 1;    //*mark[j]=i;
            }
        }
    }
}

            /*Prime Factorzation*/
//by doing some change of the General sieve code we can find the prime factor of a number
//factor(60) ---> 2 2 3 5
void factor(ll n){
    if(mark[n]==1)return ; //here change is mark array is "int" type instead of bool in general sieve
    cout<<mark[n]<<" ";
    factor(n/mark[n]);
}

             /// *** Segmented Sieve [ sqrt(up) + prime sieve ]

/// [l = lower limit, u = upper limit]
/// [first generate all prime upto sqrt(upper limit)]
/// [Checking prime
/// n = number into that segment]
if(!mark[n-l]) then it is prime
bool mark[u-l];
void segsiv(ll l, ll u)
{
    ll i,j,lt;
    if(l==1) mark[0] = 1;
    for(i=1 ; i<=in && (pr[i]*pr[i])<=u ; i++){
        lt = l/pr[i];
        lt *= pr[i];
        if(lt<l) lt += pr[i];
        if(pr[i]==lt) lt += pr[i];
        for(lt; lt<=u; lt+=pr[i]){
            mark[lt-l] = 1;
        }
    }
}

            /*NOD*/
///if n = (a1^p1) * (a2^p2) *…….*(an^pn),
///then NOD=(p1+1) * (p2+1) *….*(pn+1).
int NOD(int n) {
    // sieve method for prime calculation
    bool hash[n + 1];
    memset(hash, true, sizeof(hash));
    for (int p = 2; p * p < n; p++)
        if (hash[p] == true)
            for (int i = p * 2; i < n; i += p)
                hash[i] = false;
    // Traversing through all prime numbers
    int total = 1;
    for (int p = 2; p <= n; p++) {
        if (hash[p]) {
            int count = 0;
            if (n % p == 0) {
                while (n % p == 0) {
                    n = n / p;
                    count++;
                }
                total = total * (count + 1);
            }
        }
    }
    return total;
}


            /*SOD*/
///
prime[++nprime]=2;
    for(i=3;i<=n;i++)
    {
        if(!mark[i])
        {
            prime[++nprime]=i;

            if(i<=limit)
            {
                for(j=i*i;j<=n;j+=i*2)
                {
                    mark[j] = 1;
                }
            }
        }
    }
///  *** Disjoint Set Union Find [n||1]
int parent(int n)
{
    if(rp[n]==n)return n;
    return rp[n]=parent(rp[n]);
}
void setUp(int a,int b){
    rp[parent(b)]=parent(a);
}

            /*Generate All Subset*/

#define MX 15       //2^16==65536
int setA[MX] = {2,3,4,5};
vector <int> SumOfSubset;
void subsetSum(int position, int sum, int size)
{
    if(position == size) {SumOfSubset.push_back(sum) ; return;}
    subsetSum(position+1, sum+setA[position], size);
    subsetSum(position+1, sum, size);
    return;
}
int main()
{
    int size = sizeof(setA)/sizeof(setA[0]);
    subsetSum(0,0,size);
    return 0;
}
            /*Matrix Exponentiation for Fibonacci number*/
void power(int F[2][2], int n)
{
  if( n == 0 || n == 1)return;
  int M[2][2] = {{1,1},{1,0}};
  power(F, n/2);
  multiply(F, F);
  if (n%2 != 0)multiply(F, M);
}
void multiply(int F[2][2], int M[2][2])
{
  ///matrix multiplication and data strore into F again.
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
        F[i][j]+=F[i][j]*B[j][i];
}
int fib(int n){
  int F[2][2] = {{1,1},{1,0}};
  if(n == 0)return 0;
  power(F, n-1);
  return F[0][0];
}
            /*Find a^b*/
int exp(int a,int b){
	if(b == 1)return a;
	if(b%2!=0)return a*exp(a,b-1);
	else{ int x = exp(a,b/2); return x*x;}
}


                    /*#(2)Graph Theory Realated code*/

        /*BFS for 1d Grid*/
int node, edge;
vector <int> adj[1000];
int level[1000];
bool visited[1000];
int parent[1000];

void bfs(int source)
{
    MS(visited,0);
    MS(level,0);

    queue <int> Q;
    level[source] = 0;
    Q.push(source);
    visited[source] = 1;

    while(!Q.empty())
    {
        int u = Q.front();
        Q.pop();
        for(int i=0; i<adj[u].size(); i++)
        {
            if(visited[adj[u][i]]==0)
            {
                level[adj[u][i]] = level[u]+1;
                Q.push(adj[u][i]);
                parent[adj[u][i]] = u;
                visited[adj[u][i]] = 1;
            }
        }
    }
    return;
}

void graph_input()
{
    int x,y;
    scanf("%d%d",&node,&edge);
    for(int i=1 ; i<=edge; i++)
    {
        scanf("%d%d",&x,&y);
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
}

void ShowPath(int source, int destiny){
    cout << "Path: "<<destiny;
    for(int i=destiny; parent[i] != source;){
        cout<<" -> "<<parent[i];
        i = parent[i];
    }
    cout<<" -> "<<source<< endl;
    return;
}


int main(){
    int s=1;
    graph_input();
    bfs(4);
    ShowPath(4,1);
    return 0;
}

            /*BFS for 2D Grid*/
int fr[] = {1,-1,0,0};
int fc[] = {0,0,1,-1};
int r,c;
int cell[100][100];
int level[100][100];
bool visited[100][100];
struct node{
    int row; int col;
};

void bfs(int sr, int sc){
    node n;
    n.row = sr;
    n.col = sc;
    queue <node> Q;
    level[sr][sc] = 0;
    visited[sr][sc] = 1;
    Q.push(n);
    while(!Q.empty()){
        node u = Q.front();
        Q.pop();
        for(int i=0; i<4; i++){
            int nr=u.row+fr[i];
            int nc=u.col+fc[i];

            if(valid(nr,nc,r,c) && cell[nr][nc] != -1 && visited[nr][nc] == 0){
                node t;
                t.row = nr;
                t.col = nc;

                Q.push(t);
                level[nr][nc] = level[u.row][u.col]+1;
                visited[nr][nc] = 1;
            }
        }
    }
    return;
}

int main(){
    //scanf("%d%d",&r,&c);
    while(1){
    scanf("%d%d",&r,&c);
    bfs(0,0);
    cout <<level[3][6]<<endl;
    }
    return 0;
}
/// *** DFS [E+V]
vector<int>ed[MX];
bool vs[MX];
int lev[MX];
void dfs(int n){
    if(vs[n]) return;
    vs[n] = 1;
    int i,v;
    for(i=0;i<ed[n].size();i++){
        v = ed[n][i];
        dfs(v);
    }
}


                /*DFS{Directed, Recursuion, Cylce Deteection}*/
int node, edge,from,to;
vector <int> adj[10010];
char color[10010];
bool visited[10010];
bool is_cyclic;

void dfs(int source){
    color[source] = 'G';
    visited[source] = 1;

    for(int i=0; i<adj[source].size(); i++){
        if(color[adj[source][i]] == 'G') is_cyclic = true;
        else if(color[adj[source][i]] == 'W') dfs(adj[source][i]);
    }
    color[source] = 'B';
    return;
}

int main()
{
    FAST;
    is_cyclic = false;
    for(int i=0; i<10010; i++) {adj[i].clear(); visited[i] = 0; color[i] = 'W';}

    cin >> node >> edge;
    for(int i=0; i<edge; i++){
        cin >> from >> to;
        adj[from].push_back(to);
    }

    for(int i=1; i<=node; i++){
        if(visited[i] == 0) dfs(i);
        if(is_cyclic == true) break;
    }

    if(is_cyclic == true) printf("Is Cyclic\n");
    else printf("Not Cyclic\n");
    return 0;
}

                /*DFS (Topsort)*/
struct topsort
{
    int node;
    int ftime;
    topsort() {};
    topsort(int node_, int ftime_) {node = node_; ftime = ftime_;}
};
bool compare(topsort A, topsort B) {return A.ftime < B.ftime;}
int node, edge, from, to;
vector <int> adj[10010];
char color[10010];
int d[10010]; //discovery time
int f[10010]; //finishing time
int parent[10010];
int tme;
void dfs(int source){
    color[source] = 'G';
    tme++;
    d[source] = tme;

    for(int i=0; i<adj[source].size(); i++){
        if(color[adj[source][i]] == 'W')  {parent[adj[source][i]]=source; dfs(adj[source][i]);}
    }
    color[source] = 'B';
    tme++;
    f[source] = tme;
    return;
}

int main()
{
    FAST;
    tme = 0;
    for(int i=0; i<10010; i++){
        adj[i].clear();
        color[i] = 'W';
        d[i] = -1;
        f[i] = -1;
        parent[i] = -1;
    }
    cin >> node >> edge;
    for(int i=0; i<edge; i++){
        cin >> from >> to;
        adj[from].push_back(to);
    }

    for(int i=1; i<=node; i++){
        if(color[i] == 'W') dfs(i);
    }

    vector <topsort> nodes;
    for(int i=1; i<=node; i++){
        nodes.push_back(topsort(i,f[i]));
    }
    sort(nodes.begin(),nodes.end(),compare);

    //printing all sorted nodes
    for(int i=node-1; i>=0; i--)
    {
        cout<<nodes[i].node<<endl;
    }

    return 0;
}
                        /*DFS{Undirected , Recursion , Cycle Detection}*/

vector <int> adj[10010];
char color[10010];
int parent[10010];
bool is_cyclic;
void dfs(int source)
{
    color[source] = 'G';

    for(int i=0; i<adj[source].size(); i++)
    {
        if(color[adj[source][i]] == 'G' && parent[source] != adj[source][i]) is_cyclic = true;
        else if(color[adj[source][i]] == 'W')  {parent[adj[source][i]] = source; dfs(adj[source][i]);}
    }

    color[source] = 'B';
    return;
}

int main()
{
    FAST;
    is_cyclic = false;
    for(int i=0; i<10010; i++) {adj[i].clear(); color[i]='W'; parent[i]=0;}
    int edge,x,y;
    cin >> edge;
    for(int i=0; i<edge; i++){
        cin >>x>>y;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    parent[1] = 1;
    dfs(1);

    if(is_cyclic == true) printf("Is Cyclic\n");
    else printf("Not Cyclic\n");
    return 0;
}

///  *** Dijkstra [edge log (node)]
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
vector <int> ed[MX],ec[MX];
int ds[MX];
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
        for(i=0;i<ed[u].size();i++){
            v=ed[u][i];
            if(ds[v]>ds[u]+ec[u][i]){
                ds[v]=ds[u]+ec[u][i];
                q.push(node(v,ds[v]));
            }
        }
    }
}
                /*Floyed Warshal*/
LL node,edge;
LL cost[110][110];
void floyd()
{
    for(int k=1; k<=node; k++)
        for(int i=1; i<=node; i++)
            for(int j=1; j<=node; j++)
                if( cost[i][j] > (cost[i][k] + cost[k][j])) cost[i][j] = (cost[i][k] + cost[k][j]);
    return;
}
int main()
{
    for(int i=0; i<110; i++){
        for(int j=0; j<110; j++){
            if(i==j) cost[i][j] = 0;
            else cost[i][j] = INF;
        }
    }
    int from,to;
    LL weight;
    cin >> node >> edge;
    for(int i=0; i<edge; i++){
        cin>>from>>to>>weight;
        cost[from][to] = weight;
        cost[to][from] = weight;
    }
    floyd();
    cout << cost[1][node] <<endl;
    return 0;
}

                        /*PMSTree*/

#include <bits/stdc++.h>
using namespace std;
#define MS(ar) memset(ar,0,sizeof(ar))
struct EDGE{
    int from; int to; int cost;
    EDGE(){};
    EDGE(int _from, int _to, int _cost){
        from = _from; to = _to; cost = _cost;
    }
};
bool operator < (EDGE A, EDGE B){
    return A.cost > B.cost;
}
void show_tree();
int vis[1003], sow, node, edg;
vector <EDGE> edge[1003];
vector <EDGE> tree[1003];
void PMST(int source);
int main()
{
    ///freopen("test.txt","r",stdin);
    int t, tc=0;
    scanf("%d", &t);
    while(t--)
    {
        for(int i=0; i<100; i++) {edge[i].clear(); tree[i].clear();}
        scanf("%d %d", &node, &edg);
        int u, v, w;
        for(int i=0 ; i<edg ; i++){
            scanf("%d %d %d", &u, &v, &w);
            edge[u].push_back(EDGE(u, v, w));
            edge[v].push_back(EDGE(v, u, w));
        }
        PMST(1);
        printf("Case %d: %d\n",++tc, sow);
        show_tree();
    }
    return 0;
}
void PMST(int source){
    MS(vis);
    sow=0;
    priority_queue< EDGE >Q;
    for(int i=0 ; i<edge[source].size() ; i++){
        Q.push( edge[source][i] );
    }
    vis[source] = 1;
    while(!Q.empty()){
        EDGE f = Q.top();
        Q.pop();
        if( vis[f.to] ) continue;
        vis[f.to] = 1;
        sow += f.cost;

        tree[f.from].push_back(f);
        tree[f.to].push_back( EDGE(f.to, f.from, f.cost) );

        for(int i=0 ; i<edge[f.to].size() ; i++){
            Q.push( edge[f.to][i] );
        }
    }
    return;
}
void show_tree(){
        for(int i=0 ; i<node ; i++)
            for(int j=0 ; j<tree[i].size() ; j++)
                printf("%d %d %d\n", tree[i][j].from, tree[i][j].to, tree[i][j].cost);
        printf("\n");
    return;
}
                /*Kruskal MSTree*/
#include<bits/stdc++.h>
using namespace std;
#define ll long long
struct EDGE{
    int s;int d;int w;
};
bool compare(EDGE E1, EDGE E2){
    return E1.w < E2.w;
}
int findPar(int v, int *par)
{
    if(par[v]==v) return v;
    return findPar(par[v], par);

}
void kruskals(EDGE edg[106], int n, int E);
int main()
{
    EDGE edge[106];
    int n, E;
    cin>>n >> E;
    for(int i=0 ; i<E ; i++){
        int S, D, W;
        cin>> S>>D>>W;
        edge[i].s = S;
        edge[i].d = D;
        edge[i].w = W;
    }
    kruskals(edge, n, E);
    return 0;
}
void kruskals(EDGE edg[106], int n, int E)
{
    sort(edg, edg+E, compare);
    EDGE edgo[106];
    int count=0, i=0;
    int parent[106];
    for(int i=0; i<n ; i++)parent[i]=i;
    while(count != n-1){
        EDGE curedge = edg[i];
        int sourcePar = findPar(curedge.s, parent);
        int destPar = findPar(curedge.d, parent);
        if(sourcePar != destPar){
            edgo[count] = curedge;
            count++;
            parent[sourcePar] = destPar;
        }
        i++;
    }
    for(int i=0; i<n-1 ; i++) cout<< edgo[i].s << " "<<edgo[i].d<<" "<<edgo[i].w<<endl;
}
                    /*#Data Structure*/

                /*Segment Tree*/

///(i)Costruct (ii)Getsum (iii)Update
#include <bits/stdc++.h>
using namespace std;
int getMid(int s, int e) {  return s + (e -s)/2;  }
void updateValueUtil(int *st, int ss, int se, int i, int diff, int si)
{
    if (i < ss || i > se)return;
    st[si] = st[si] + diff;
    if (se != ss){
        int mid = getMid(ss, se);
        updateValueUtil(st, ss, mid, i, diff, 2*si + 1);
        updateValueUtil(st, mid+1, se, i, diff, 2*si + 2);
    }
}
void updateValue(int arr[], int *st, int n, int i, int new_val)
{
    if (i < 0 || i > n-1){
        printf("Invalid Input");
        return;
    }
    int diff = new_val - arr[i];
    arr[i] = new_val;
    updateValueUtil(st, 0, n-1, i, diff, 0);
}
int getSumUtil(int *st, int ss, int se, int qs, int qe, int si)
{
    if (qs <= ss && qe >= se)return st[si];

    if (se < qs || ss > qe) return 0;
    int mid = getMid(ss, se);
    return getSumUtil(st, ss, mid, qs, qe, 2*si+1) +
           getSumUtil(st, mid+1, se, qs, qe, 2*si+2);
}
int getSum(int *st, int n, int qs, int qe)
{
    if (qs < 0 || qe > n-1 || qs > qe){
        printf("Invalid Input");
        return -1;
    }
    return getSumUtil(st, 0, n-1, qs, qe, 0);
}
int constructSTUtil(int arr[], int ss, int se, int *st, int si)
{
    if (ss == se){
        st[si] = arr[ss];
        return arr[ss];
    }
    int mid = getMid(ss, se);
    st[si] =  constructSTUtil(arr, ss, mid, st, si*2+1) +
              constructSTUtil(arr, mid+1, se, st, si*2+2);
    return st[si];
}
int *constructST(int arr[], int n)
{
    int x = (int)(ceil(log2(n)));
    int max_size = 2*(int)pow(2, x) - 1;

    int *st = new int[max_size];
    constructSTUtil(arr, 0, n-1, st, 0);

    for(int i=0; i<max_size ; i++)cout<<st[i]<<" ";
    cout<<endl;
    return st;
}
int main()
{
    int arr[] = {1, 3, 5, 7, 9, 11};
    int n = sizeof(arr)/sizeof(arr[0]);

    int *st = constructST(arr, n);  //0 based index
    printf("Sum of values in given range = %d\n", getSum(st, n, 1, 3));

    updateValue(arr, st, n, 1, 10);
    printf("Updated sum of values in given range = %d\n", getSum(st, n, 1, 3));
    return 0;
}

///                 *** Segment Tree [log(total array size)*Query]

/// [ulow,uhigh] Query Range
/// [low,high] total range of root
/// Currrent position = pos
/// 0 based Index And Root is also 0


int ara[MX],seg[4*MX],lazy[4*MX];

void creat(int low,int high,int pos)
{
    if(low==high){
        seg[pos] = ara[low]; // reached leaf and update
        return ;
    }
    int mid = (high+low)/2;
    creat(low,mid,pos*2+1);
    creat(mid+1,high,pos*2+2);
    seg[pos] += seg[pos*2+1] + seg[pos*2+2];
}

void update(int low,int high,int ulow,int uhigh,int val,int pos)
{
    if(low>high) return ;
    if(lazy[pos]!=0){ /// is not propagated yet
        seg[pos] = 0;
        if(low!=high){  ///if not leaf node
            lazy[pos*2+1] += lazy[pos];
            lazy[pos*2+2] += lazy[pos];
        }
        lazy[pos] = 0;
    }

    if(ulow>high||uhigh<low) return; ///No overlap
    if(ulow<=low&&uhigh>=high){ /// Total Overlap
        seg[pos] += val;
        if(low!=high){
            lazy[pos*2+1] += val;
            lazy[pos*2+2] += val;
        }
        return;
    }
    /// Partial overlap
    int mid = (high+low)/2;

    update(low,mid,ulow,uhigh,val,pos*2+1);
    update(mid+1,high,ulow,uhigh,val,pos*2+2);
    seg[pos] = seg[pos*2+1] + seg[pos*2+2]; /// Updating the intermediate node
}

int query(int low,int high,int ulow,int uhigh,int pos)
{
    if(low>high) return 0;
    if(lazy[pos]!=0){
        seg[pos] += lazy[pos];
        if(low!=high){
            lazy[pos*2+1] += lazy[pos];
            lazy[pos*2+2] += lazy[pos];
        }
        lazy[pos] = 0;
    }

    if(ulow>high||uhigh<low) return 0;

    if(ulow<=low&&uhigh>=high)
        return seg[pos];

    int mid = (high+low)/2;

    return query(low,mid,ulow,uhigh,pos*2+1) + query(mid+1,high,ulow,uhigh,pos*2+2);
}




                                ///*General Useful Technique*////
/*
    #Base Conversion::> log(x)(N)/ log(x)(b) //x=current base of N.  b=desired base
    #Number Of digits of N! is CEILING[log(N) +log(N-1) ... +log(1)]
    #power(exp,base) func:::> while (exp){if(exp & 1)result *= base; exp >>= 1; base *= base;}

                {COMBINATORICS, TOTIENT FUNCTION, GAME THEORY, CONVEX HULL, BITMASK}

    # vector<int>:: iterator it; for(it=vt.begin(); it<vt.end(); it++){printf("%d\n", *it);}
    # sort( v.begin(), v.end() , compare); //sort by comparison criteria
    # Operator Overload(inside struct)-> bool operator < (const data& b)const{return income > b.income;}
    # sort array from 3 to n --> sort( array+3, array+n );
    # reverse( vt.begin(), vt.end() );    //not sort,just reverse the vector
//
    *Set::all data in a 'set' is unique,,(easily find how many unique data ) also sorted.
        set< int > s;   s.insert( 10 ); s.insert( 5 ); s.insert( 9 );
        set< int > :: iterator it;
        for(it = s.begin(); it != s.end(); it++) cout << *it << endl;
    *StringStream:: Given a input in online,,easily can access those number;
        (i)stringstream ss( line ); int num; vector< int > v;
        while( ss >> num ) v.push_back( num ); //num enter as a general int type number..
        (ii)stringstream ss;
        ss<<hex<<result[i]; //input as needed   //ss<<something
        buffer = ss.str();

    *next_permutation:: Generate all permutation in Sorted order
        for(int i=0; i<3; i++) v.push_back( i );      //012 021 102 120 201 210
        do{cout<<v[0]<<v[1]<<v[2]<<" ";}while( next_permutation( v.begin(), v.end() ) );
//
*/



