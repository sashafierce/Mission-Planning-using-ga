#include <bits/stdc++.h>

double distanceBetweenTwoPoints(double x, double y, double a, double b){
	using namespace std;
	return sqrt(pow(x - a, 2) + pow(y - b, 2));
}
int main() {
	using namespace std;
	int s = 100;

	freopen("coord100.txt","w+",stdout);
	vector<pair<int, int> > coord;
	while(s--){
		
		int x = rand()%2000;
		int y = rand()%2000;
		int asc = rand()%26 + 97;
		char n = asc;
		cout<<n<<" "<<x<<" "<<y<<endl;
                coord.push_back(make_pair(x,y));
	}

        freopen("transition100.txt","w+",stdout);
        for(int i =0 ; i < 100; i++) {
		 for(int j =0 ; j < 100; j++) {
			printf("%4.4f",distanceBetweenTwoPoints(coord[i].first,coord[i].second, coord[j].first,coord[j].second));
			if (j != 99)cout<<'\t';	
		}
		cout<<endl;	
	}	
	
	return 0;
}
