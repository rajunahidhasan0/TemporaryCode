#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define MX 100005
#define MOD 1000000007LL
#define MS(ARRAY,VALUE) memset(ARRAY,VALUE,sizeof(ARRAY))
#define Fin freopen("input.txt","r",stdin)
#define FAST ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

#define Fout freopen("output.txt","w",stdout)
#define rep(i,a,b) for(i=a;i<=b;i++)
#define EPS 0.00000001
#define INF INT_MAX
#define PI 2*acos(0.0)
#define P1(XX) cout<<"db1: "<<XX<<endl
#define P2(XX,YY) cout<<"db2: "<<XX<<" "<<YY<<endl
#define P3(XX,YY,ZZ) cout<<"db3: "<<XX<<" "<<YY<<" "<<ZZ<<endl
#define set(XX,POS) XX|=(1<<POS)
#define reset(XX,POS) XX&=(~(1<<POS))
#define check(XX,POS) (bool)(XX&(1<<POS))
#define toggle(XX,POS) (XX^(1<<POS))
#define SORT(v) sort(v.begin(),v.end())
#define REVERSE(V) reverse(v.begin(),v.end())
#define VALID(X,Y,R,C) X>=0 && X<R && Y>=0 && Y<C
#define SIZE(ARRAY) sizeof(ARRAY)/sizeof(ARRAY[0])
#define RT printf("Run Time : %0.3lf seconds\n", clock()/(CLOCKS_PER_SEC*1.0))


/*------------------------------Graph Moves----------------------------*/
//const int fx[]={+1,-1,+0,+0};
//const int fy[]={+0,+0,+1,-1};
//const int fx[]={+0,+0,+1,-1,-1,+1,-1,+1};   // Kings Move
//const int fy[]={-1,+1,+0,+0,+1,+1,-1,-1};  // Kings Move
//const int fx[]={-2, -2, -1, -1,  1,  1,  2,  2};  // Knights Move
//const int fy[]={-1,  1, -2,  2, -2,  2, -1,  1}; // Knights Move
/*---------------------------------------------------------------------*/

void free(){MS(dp,-1);}


int main()
{
    ll m, n, ans=0, i, j;
    cin>>n;
    for(i=0;i<n;i++){
            scanf("%lld", &a[i]);
    }

    cout<<ans<<endl;
    return 0;
}



int main()
{
    int test, tc=0;
    ll m, n, ans, i, j;
    scanf("%d", &test);
    while(test--)
    {
        ans=0;
        cin>>n;
        for(i=0 ; i<n ; i++){
                scanf("%lld", &a[i]);
        }
        printf("Case %d: %lld\n", ++tc, ans);
    }
    return 0;
}
int main()
{
    int test, tc=0;
    ll m, n, ans, i, j;
    cin>>test;
    while(test--)
    {
        ans=0;
        cin>>n;
        for( i=0 ; i<n ; i++){
            cin>>a[i];
        }
        cout<<"Case "<<++tc<<": "<<ans<<endl;
    }
    return 0;
}

